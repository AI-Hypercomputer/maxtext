# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pipeline parallelism under shard_mode=explicit.

The mesh tests run as a subprocess so the forced CPU device count takes effect before JAX initializes; the parent
pytest process has already initialized JAX with the default device count. The child runs the stage function the way
the pipeline does under explicit sharding (a vmap over 'stage' as an Auto axis, with a shard_map inside that gathers
from an unbatched table at token-dependent indices, like the MoE), and trains a pipelined DeepSeek model in both shard
modes on a pipeline x FSDP CPU mesh.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import numpy as np
import pytest

from maxtext.common.common_types import ShardMode
from maxtext.utils import pipeline_utils

_NUM_CPU_DEVICES = 8


@pytest.mark.cpu_only
def test_pipeline_explicit_sharding_on_cpu_mesh():
  """Runs the mesh tests below in a subprocess with forced CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_CPU_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "PIPELINE_EXPLICIT_SHARDING_MESH_TESTS_PASSED" in result.stdout


def _mesh(shape, axis_types):
  devices = np.asarray(jax.devices()[: int(np.prod(shape))]).reshape(shape)
  return Mesh(devices, ("data", "stage", "fsdp"), axis_types=axis_types)


class StageMeshTest(unittest.TestCase):
  """pipeline_stage_mesh and the PartitionSpec helpers; any device count."""

  def test_only_stage_becomes_auto_under_explicit(self):
    mesh = _mesh((1, 1, 1), (AxisType.Explicit,) * 3)
    stage_mesh = pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.EXPLICIT)
    self.assertEqual(stage_mesh.axis_types, (AxisType.Explicit, AxisType.Auto, AxisType.Explicit))
    self.assertEqual(stage_mesh.axis_names, mesh.axis_names)
    self.assertTrue((stage_mesh.devices == mesh.devices).all())

  def test_unchanged_under_auto(self):
    mesh = _mesh((1, 1, 1), (AxisType.Auto,) * 3)
    self.assertIs(pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.AUTO), mesh)

  def test_remove_mesh_axis(self):
    self.assertIsNone(pipeline_utils.remove_mesh_axis(None, "stage"))
    self.assertIsNone(pipeline_utils.remove_mesh_axis("stage", "stage"))
    self.assertEqual(pipeline_utils.remove_mesh_axis("fsdp", "stage"), "fsdp")
    self.assertEqual(pipeline_utils.remove_mesh_axis(("stage", "fsdp"), "stage"), ("fsdp",))
    self.assertIsNone(pipeline_utils.remove_mesh_axis(("stage",), "stage"))

  def test_remove_mesh_axis_from_spec_keeps_reduced(self):
    spec = P("stage", ("stage", "fsdp"), None, reduced={"data"})
    self.assertEqual(pipeline_utils.remove_mesh_axis_from_spec(spec, "stage"), P(None, ("fsdp",), None, reduced={"data"}))


class PipelineExplicitMeshTest(unittest.TestCase):
  """Runs on 8 forced CPU devices, in the subprocess started by the test above."""

  __test__ = False

  def test_stage_vmap_runs_shard_map_gather(self):
    """A shard_map inside the stages, gathering from an unbatched table at token-dependent indices (as the MoE does).

    vmapped over an Explicit 'stage' axis, JAX replicates the stage dim into the shard_map and cannot batch the
    gather; call_with_stage_axis_auto runs it per stage as under shard_mode=auto.
    """
    mesh = _mesh((2, 2, 2), (AxisType.Explicit,) * 3)
    stage_mesh = pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.EXPLICIT)
    table = jnp.arange(4, dtype=jnp.float32)

    def stage_fn(x, w):
      x = reshard(x, P(("data", "fsdp"), None))
      y = jnp.einsum("bd,df->bf", x, w, out_sharding=P(("data", "fsdp"), None))

      @jax.shard_map(mesh=stage_mesh, in_specs=P(("data", "fsdp"), None), out_specs=P(("data", "fsdp"), None))
      def route(y_local):
        return y_local * table[(y_local > 0).astype(jnp.int32) + (y_local > 1).astype(jnp.int32)]

      return route(y)

    def run(x, w):
      return pipeline_utils.call_with_stage_axis_auto(jax.vmap(stage_fn, spmd_axis_name="stage"), stage_mesh, x, w)

    x_host = np.random.default_rng(0).normal(size=(2, 8, 4)).astype(np.float32)
    w_host = np.random.default_rng(1).normal(size=(2, 4, 16)).astype(np.float32)

    def reference(x, w):
      y = jnp.einsum("sbd,sdf->sbf", x, w)
      return y * table[(y > 0).astype(jnp.int32) + (y > 1).astype(jnp.int32)]

    with jax.set_mesh(mesh):
      x = reshard(jnp.asarray(x_host), P("stage", ("data", "fsdp"), None))
      w = reshard(jnp.asarray(w_host), P("stage", "fsdp", None))
      out = jax.jit(run)(x, w)
      grad = jax.jit(jax.grad(lambda x, w: (run(x, w) ** 2).sum(), argnums=1))(x, w)
    self.assertEqual(out.sharding.spec, P("stage", ("data", "fsdp"), None))
    self.assertEqual(grad.sharding.spec, P("stage", "fsdp", None))
    np.testing.assert_allclose(np.asarray(out), reference(x_host, w_host), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(grad),
        jax.grad(lambda w: (reference(x_host, w) ** 2).sum())(w_host),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_deepseek_explicit_matches_auto(self):
    """A pipelined DeepSeek model trains to the same losses in both shard modes on a pipeline x FSDP CPU mesh.

    Covers the sparse-matmul MoE inside the stages, the dense layer before the pipeline and the MTP block under the
    PP-as-DP rules, and the per-layer reduction of the sown MoE metrics. In float32 on CPU the two modes agree to
    rounding.
    """
    # pylint: disable-next=import-outside-toplevel  (imports the trainer only in the 8-device subprocess)
    from maxtext.trainers.pre_train.train import main as train_main
    from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT  # pylint: disable=import-outside-toplevel
    from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

    def losses(shard_mode):
      with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_file = os.path.join(tmp_dir, "metrics.txt")
        train_main(
            [
                None,
                get_test_config_path(),
                f"base_output_directory={tmp_dir}",
                f"run_name=pp_deepseek_{shard_mode}",
                f"metrics_file={metrics_file}",
                f"shard_mode={shard_mode}",
                "dataset_type=synthetic",
                "steps=2",
                "enable_checkpointing=False",
                "enable_goodput_recording=False",
                "enable_tensorboard=False",
                rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
                "model_name=deepseek3-tiny",
                "override_model_config=True",
                "base_num_decoder_layers=5",
                "first_num_dense_layers=1",
                "vocab_size=256",
                "num_experts=4",
                "num_experts_per_tok=2",
                "mtp_num_layers=1",
                "mtp_loss_scaling_factor=0.1",
                "max_target_length=64",
                "per_device_batch_size=2",
                "dtype=float32",
                "matmul_precision=highest",
                "ici_pipeline_parallelism=2",
                "ici_fsdp_parallelism=4",
                "num_pipeline_microbatches=4",
            ]
        )
        with open(metrics_file, "rt", encoding="utf8") as f:
          return [json.loads(line)["learning/loss"] for line in f if line.strip()]

    auto_losses = losses("auto")
    explicit_losses = losses("explicit")
    self.assertEqual(len(auto_losses), 2)
    np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-5, atol=0.0)


if __name__ == "__main__":
  PipelineExplicitMeshTest.__test__ = True
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineExplicitMeshTest)
  res = unittest.TextTestRunner(verbosity=2).run(suite)
  if res.wasSuccessful() and res.testsRun == 2 and not res.skipped:
    print("PIPELINE_EXPLICIT_SHARDING_MESH_TESTS_PASSED")
  sys.exit(0 if res.wasSuccessful() else 1)

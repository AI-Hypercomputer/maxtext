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

"""Tests for moe_combine_rs_chunk_overlap: the scheduling-group annotation that overlaps chunk c-1's combine
reduce-scatter with chunk c's expert GMMs changes the instruction order only. Output, loss and gradients must be
bit-identical to the flag-off chunked ring-of-experts layer, and the annotation must only appear with the flag on."""

import os
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.layers import moe
from tests.utils.test_helpers import get_test_config_path

_REQUIRED_CPU_DEVICES = 8

_RING_CONFIG = {
    "run_name": "test",
    "num_experts": 8,
    "base_mlp_dim": 64,
    "base_moe_mlp_dim": 64,
    "override_logical_axis_rules": True,
    "use_ring_of_experts": True,
    "use_ragged_sort": True,
    "ragged_buffer_factor": 1.5,
}


class ConfigValidationTest(unittest.TestCase):
  """moe_combine_rs_chunk_overlap needs the chunked ring-of-experts path."""

  def test_requires_token_chunks(self):
    with self.assertRaisesRegex(ValueError, "moe_combine_rs_chunk_overlap requires num_moe_token_chunks > 1"):
      types.MaxTextConfig(**_RING_CONFIG, moe_combine_rs_chunk_overlap="rs")

  def test_accepts_chunked_ring_of_experts(self):
    for mode in ("rs", "unpermute_rs"):
      config = types.MaxTextConfig(**_RING_CONFIG, moe_combine_rs_chunk_overlap=mode, num_moe_token_chunks=2)
      self.assertEqual(config.moe_combine_rs_chunk_overlap, mode)

  def test_rejects_unknown_mode(self):
    with self.assertRaises(ValueError):
      types.MaxTextConfig(**_RING_CONFIG, moe_combine_rs_chunk_overlap="bogus", num_moe_token_chunks=2)


@pytest.mark.cpu_only
def test_combine_rs_chunk_overlap_on_cpu_mesh():
  """Runs the mesh test below in a subprocess with 8 forced CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_REQUIRED_CPU_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "COMBINE_RS_CHUNK_OVERLAP_MESH_TESTS_PASSED" in result.stdout


class CombineOverlapMeshTest(unittest.TestCase):
  """The real RoutedMoE ring-of-experts layer (EP=2, fsdp=4, 2 token chunks, jax.jit on CPU)."""

  __test__ = False

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < _REQUIRED_CPU_DEVICES:
      self.skipTest("needs 8 CPU devices; run through test_combine_rs_chunk_overlap_on_cpu_mesh")
    if not hasattr(xla_metadata, "xla_metadata_call2"):
      self.skipTest("moe_combine_rs_chunk_overlap needs jax.experimental.xla_metadata.xla_metadata_call2")

  def _run(self, mode, chunks=2, rbf=-1.0, combine_bwd_method=""):
    """Returns (output, loss, grads, lowered HLO text) of the layer's loss and grad under jax.jit."""
    # pylint: disable=import-outside-toplevel
    from flax.linen import partitioning as nn_partitioning
    from maxtext.layers.initializers import nd_dense_init
    from maxtext.utils import maxtext_utils

    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="combine_rs_chunk_overlap",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        override_model_config=True,
        base_emb_dim=256,
        base_mlp_dim=128,
        base_moe_mlp_dim=128,
        dtype="bfloat16",
        weight_dtype="float32",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=2,
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        max_target_length=64,
        use_ragged_sort=True,
        ragged_buffer_factor=rbf,
        num_moe_token_chunks=chunks,
        moe_combine_rs_chunk_overlap=mode,
        moe_quantize_combine_bwd_method=combine_bwd_method,
        routed_bias=True,
        routed_bias_update_rate=0.01,
        routed_score_func="sigmoid",
        decoder_block="deepseek",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )
    batch = int(cfg.per_device_batch_size) * jax.device_count()
    x = jax.random.normal(jax.random.PRNGKey(1), (batch, cfg.max_target_length, cfg.base_emb_dim), cfg.dtype)
    rng = jax.random.PRNGKey(0)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables = jax.jit(lambda h: model.init({"params": rng, "dropout": rng}, h))(x)

      def loss_fn(p, h):
        out, _, _ = model.apply({"params": p}, h)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

      f = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
      hlo = f.lower(variables["params"], x).as_text()
      (loss, out), grads = f(variables["params"], x)
    return out, loss, grads, hlo

  def _check_modes_match_flag_off(self, combine_bwd_method=""):
    """Both overlap modes reproduce the flag-off output, loss and gradients exactly and only add scheduling groups."""
    out_ref, loss_ref, grads_ref, hlo_ref = self._run("", combine_bwd_method=combine_bwd_method)
    self.assertNotIn("_scheduling_group_id", hlo_ref)
    for mode in ("rs", "unpermute_rs"):
      with self.subTest(mode=mode):
        out, loss, grads, hlo = self._run(mode, combine_bwd_method=combine_bwd_method)
        # Ordering only: same ops, same numbers.
        np.testing.assert_array_equal(np.asarray(out), np.asarray(out_ref))
        np.testing.assert_array_equal(np.asarray(loss), np.asarray(loss_ref))
        for g, g_ref in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(grads_ref)):
          np.testing.assert_array_equal(np.asarray(g), np.asarray(g_ref))
        # Scheduling groups are present (ids start at 20000 so they cannot collide with other schedules).
        self.assertIn("_scheduling_group_id", hlo)
        for gid in sorted({int(s.split('"')[0]) for s in hlo.split('_scheduling_group_id = "')[1:]}):
          self.assertGreaterEqual(gid, 20_000)

  def test_modes_match_flag_off(self):
    self._check_modes_match_flag_off()

  def test_modes_match_flag_off_quantized_combine_bwd(self):
    """The custom-VJP reduce-scatter (moe_quantize_combine_bwd_method) also traces inside the scheduling group."""
    self._check_modes_match_flag_off(combine_bwd_method="rowwise")


if __name__ == "__main__":
  CombineOverlapMeshTest.__test__ = True
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(CombineOverlapMeshTest)
  res = unittest.TextTestRunner(verbosity=2).run(suite)
  if res.wasSuccessful() and res.testsRun == 2 and not res.skipped:
    print("COMBINE_RS_CHUNK_OVERLAP_MESH_TESTS_PASSED")
  sys.exit(0 if res.wasSuccessful() else 1)

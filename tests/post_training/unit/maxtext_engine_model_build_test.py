# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The engine builds its model whatever mesh the caller has entered.

`with jax.set_mesh(mesh): MaxTextTrainingEngine(cfg, mesh=mesh)` is how `train.py`-style code
and this engine's own constructor and data-parallel tests construct it. Under that ambient mesh,
flax's `nnx.eval_shape` inside `create_nnx_abstract_model` re-derives every variable's sharding
from its logical names, and raises for a name the rules leave unmapped -- e.g.
`Resource axis: norm of P('norm',) is not found in mesh` under the `cp-as-ep` rules. The tiny
dense configs the other tests use have no such name.

So this builds a shrunk Qwen3.5-397B-A17B -- the MoE, GDN and ring-attention layers, the
`cp-as-ep` rules, a mesh with fsdp, context and expert all above 1 -- under each of the three
ways a caller can hold a mesh. Construction only: the Tokamax, GDN and megablox kernels are
TPU-only, and the mesh matters only while the model is built.

It runs in a subprocess for the reason `maxtext_engine_data_parallel_test.py` gives: the CPU
backend fixes its device count at initialization, which a sibling module has already done by
the time pytest imports this file.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

from absl.testing import absltest
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 8
_SENTINEL = "MAXTEXT_ENGINE_MODEL_BUILD_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

# The sharding and kernel flags of a Qwen3.5-397B-A17B training configuration, on a model shrunk to
# build in seconds and an fsdp2 x context2 x expert2 mesh.
_TINY_QWEN35_397B_CONFIG = (
    "model_name=qwen3.5-397b-a17b",
    "override_model_config=true",
    "custom_mesh_and_rule=cp-as-ep",
    "ici_tensor_parallelism=1",
    "ici_fsdp_parallelism=2",
    "ici_context_parallelism=2",
    "ici_expert_parallelism=2",
    "context_parallel_strategy=ring",
    "context_parallel_load_balance=False",
    "allow_split_physical_axes=False",
    "attention=flash",
    "use_tokamax_splash=true",
    "use_splash_scheduler=true",
    "use_gdn_kernel=true",
    "gdn_cp_mode=auto",
    "gdn_chunk_size=64",
    "megablox=true",
    "sparse_matmul=true",
    "use_tokamax_gmm=true",
    "use_gmm_v2=true",
    "use_ring_of_experts=true",
    "use_ragged_sort=true",
    "remat_policy=custom",
    "decoder_layer_input=device",
    "context=device",
    "gdn=remat",
    "gdn_conv=remat",
    "num_vocab_tiling=2",
    "dtype=bfloat16",
    "mu_dtype=bfloat16",
    "opt_type=adamw",
    "scan_layers=True",
    "packing=True",
    "enable_gdn_sequence_packing=True",
    "optimizer_memory_host_offload=true",
    "grad_dtype=float32",
    "enable_checkpointing=false",
    "dataset_type=synthetic",
    "base_num_decoder_layers=4",
    "base_emb_dim=256",
    "base_num_query_heads=4",
    "base_num_kv_heads=2",
    "head_dim=256",
    "base_mlp_dim=128",
    "base_moe_mlp_dim=128",
    "num_experts=8",
    "num_experts_per_tok=2",
    "shared_experts=1",
    "gdn_key_head_dim=32",
    "gdn_value_head_dim=32",
    "gdn_num_key_heads=4",
    "gdn_num_value_heads=8",
    "vocab_size=1024",
    "max_target_length=1024",
    "per_device_batch_size=0.25",
    "use_multimodal=false",
    "skip_jax_distributed_system=True",
    "enable_tensorboard=False",
    "convert_checkpoint_if_possible=False",
)


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_model_build_on_eight_cpu_devices():
  """The only test pytest collects here; the class below runs inside the child process."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # An exit status of 0 is also what a child that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


class ModelBuildTest(absltest.TestCase):
  """Construction of the tiny Qwen3.5-397B config, under each way a caller can hold a mesh."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  @classmethod
  def setUpClass(cls):
    """Builds the config and its mesh once; the tests differ only in how the engine is constructed."""
    super().setUpClass()
    cls.config = pyconfig.initialize(
        [
            "maxtext_engine_model_build_test.py",
            get_test_config_path("base.yml"),
            "run_name=engine_model_build_test",
            f"base_output_directory={tempfile.mkdtemp()}",
            *_TINY_QWEN35_397B_CONFIG,
        ]
    )
    cls.mesh = maxtext_utils.get_mesh_from_config(cls.config)

  def _assert_built_on(self, engine, mesh):
    self.assertIsInstance(engine.model, nnx.Module)
    self.assertEqual(dict(engine._mesh.shape), dict(mesh.shape))  # pylint: disable=protected-access

  def test_config_leaves_norm_unmapped(self):
    """Without an unmapped logical name (`norm`) the tests below could not fail."""
    self.assertEqual(self.config.custom_mesh_and_rule.value, "cp-as-ep")
    self.assertEqual({k: v for k, v in self.mesh.shape.items() if v > 1}, {"fsdp": 2, "context": 2, "expert": 2})
    self.assertNotIn("norm", {rule[0] for rule in self.config.logical_axis_rules})

  def test_build_under_set_mesh(self):
    """A caller's `jax.set_mesh` is cleared while the model is built, then restored."""
    with jax.set_mesh(self.mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(self.config, mesh=self.mesh)
      # Cleared for the build only: the caller's mesh is back once the engine exists.
      self.assertEqual(jax.sharding.get_abstract_mesh(), self.mesh.abstract_mesh)
    self._assert_built_on(engine, self.mesh)

  def test_build_under_legacy_mesh_context(self):
    """Tunix's `create_maxtext_engine` enters only `with mesh:`, which does not set the abstract mesh."""
    with self.mesh:
      engine = maxtext_engine.MaxTextTrainingEngine(self.config, mesh=self.mesh)
    self._assert_built_on(engine, self.mesh)

  def test_build_without_mesh(self):
    """With no mesh given, the engine adopts the one `from_pretrained` derives."""
    engine = maxtext_engine.MaxTextTrainingEngine(self.config)
    self._assert_built_on(engine, self.mesh)


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelBuildTest))
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")

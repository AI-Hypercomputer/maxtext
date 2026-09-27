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

"""Resuming from an intra-step checkpoint, through real Orbax, gives the uninterrupted step bit for bit.

A step of G micro-batches can be checkpointed part-way: `save_checkpoint` writes the gradients
summed so far, and a fresh engine restores them and finishes the step. The gradients are summed
in the accumulation dtype (`grad_accumulation_dtype`, else `grad_dtype`), which need not be the
parameters' dtype, and Orbax restores each leaf in its target's dtype. So the restore target
must carry the accumulation dtype: otherwise a bf16 sum under float32 weights comes back as
float32, and a float32 sum under bf16 weights is rounded to bf16.

Orbax is not mocked here. Each case runs the step three ways -- uninterrupted; stopped after two
of three micro-batches and saved; restored into a fresh engine and finished -- and requires the
last to match the first exactly, both with `compile(dummy)` and on the deferred `compile(None)`
path Tunix's `TrainerWorker` takes. Cases whose accumulator dtype equals the weights' dtype are
included alongside the mixed ones.

Runs in a subprocess on a four-device CPU mesh, so the parameters (and therefore the restore
target) are genuinely sharded; see `maxtext_engine_data_parallel_test.py` for why a subprocess.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_CHECKPOINT_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")
_MICRO_BATCHES = 3
# Saved after this many micro-batches, so the saved accumulator is already a sum, not one gradient.
_SAVED_AFTER = 2


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_checkpoint_resume_on_four_cpu_devices():
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


def _config(checkpoint_dir: str, **overrides) -> pyconfig.HyperParameters:
  """A tiny dense model with plain SGD, so every difference in the gradient reaches the weights."""
  fields = {
      "model_name": "default",
      "run_name": "engine_checkpoint_test",
      "base_output_directory": tempfile.mkdtemp(),
      "vocab_size": 128,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "max_target_length": 32,
      "per_device_batch_size": 1,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "enable_dropout": False,
      "enable_tensorboard": False,
      "record_internal_nn_metrics": False,
      "skip_jax_distributed_system": True,
      "convert_checkpoint_if_possible": False,
      "profiler_steps": 0,
      "log_config": False,
      "init_weights_seed": 0,
      "opt_type": "sgd",
      "learning_rate": 0.1,
      "warmup_steps_fraction": 0.0,
      "learning_rate_final_fraction": 1.0,
      "gradient_clipping_threshold": 0.0,
      "remat_policy": "none",
      "gradient_accumulation_steps": _MICRO_BATCHES,
      "enable_checkpointing": True,
      "checkpoint_dir": checkpoint_dir,
      "async_checkpointing": False,
      "checkpoint_period": 1,
  }
  fields.update(overrides)
  argv = ["maxtext_engine_checkpoint_test.py", get_test_config_path("base.yml")]
  return pyconfig.initialize(argv + [f"{key}={value}" for key, value in fields.items()])


def _micro_batches(cfg: pyconfig.HyperParameters) -> list[dict[str, np.ndarray]]:
  """`_MICRO_BATCHES` language-model micro-batches with ragged rows, so no loss denominator is a power of two."""
  rows, seq, vocab = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length, cfg.vocab_size
  rng = np.random.default_rng(0)
  batches = []
  for m in range(_MICRO_BATCHES):
    lengths = seq - (np.arange(rows) * 3 + m) % 7
    real = (np.arange(seq)[None, :] < lengths[:, None]).astype(np.int32)
    tokens = rng.integers(1, vocab, size=(rows, seq)).astype(np.int32) * real
    batches.append(
        {
            "inputs": tokens,
            "targets": np.roll(tokens, -1, axis=-1) * np.roll(real, -1, axis=-1),
            "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (rows, 1)),
            "inputs_segmentation": real,
            "targets_segmentation": real * np.roll(real, -1, axis=-1),
        }
    )
  return batches


def _params(engine) -> dict[str, np.ndarray]:
  leaves = jax.tree_util.tree_leaves_with_path(nnx.to_pure_dict(nnx.state(engine.state.model, nnx.Param)))
  return {jax.tree_util.keystr(path): np.asarray(leaf, np.float64) for path, leaf in leaves}


def _host_leaves(tree) -> list[np.ndarray]:
  return [np.asarray(leaf, np.float64) for leaf in jax.tree.leaves(tree)]


class IntraStepCheckpointTest(parameterized.TestCase):
  """Stop after two of three micro-batches, save, restore into a fresh engine, finish: same weights."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def _engine(self, cfg, mesh, mode, batch):
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    engine.compile(batch if mode == "dummy" else None)
    return engine

  @parameterized.product(
      (
          # float32 weights, gradients summed in bf16.
          {"overrides": {"grad_dtype": "bfloat16"}, "accumulation_dtype": jnp.bfloat16},
          # bf16 weights, gradients summed in float32 because `grad_dtype` is float32.
          {"overrides": {"weight_dtype": "bfloat16", "grad_dtype": "float32"}, "accumulation_dtype": jnp.float32},
          # bf16 weights, gradients summed in float32 through `grad_accumulation_dtype`.
          {
              "overrides": {"weight_dtype": "bfloat16", "grad_dtype": "bfloat16", "grad_accumulation_dtype": "float32"},
              "accumulation_dtype": jnp.float32,
          },
          # Accumulator dtype equal to the weights' dtype.
          {"overrides": {"grad_dtype": "float32"}, "accumulation_dtype": jnp.float32},
          {
              "overrides": {"grad_dtype": "bfloat16", "grad_accumulation_dtype": "float32"},
              "accumulation_dtype": jnp.float32,
          },
      ),
      # `compile(dummy)` builds executables up front; `compile(None)` is Tunix's deferred path.
      mode=("dummy", "deferred"),
  )
  def test_resumed_step_matches_uninterrupted(self, overrides, accumulation_dtype, mode):
    cfg = _config(tempfile.mkdtemp(), **overrides)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    batches = _micro_batches(cfg)
    self.assertGreater(len(jax.devices()), 1, "the restore target is only sharded on a multi-device mesh")

    uninterrupted = self._engine(cfg, mesh, mode, batches[0])
    for batch in batches:
      uninterrupted.fwd_bwd(batch)
    uninterrupted.update()

    stopped = self._engine(cfg, mesh, mode, batches[0])
    for batch in batches[:_SAVED_AFTER]:
      stopped.fwd_bwd(batch)
    saved = _host_leaves(stopped._reduced_accumulated_grads())  # pylint: disable=protected-access
    self.assertEqual(
        {leaf.dtype for leaf in jax.tree.leaves(stopped._accumulated_grads)},  # pylint: disable=protected-access
        {jnp.dtype(accumulation_dtype)},
    )
    stopped.save_checkpoint(metadata={"case": mode})
    stopped._checkpoint_manager.wait_until_finished()  # pylint: disable=protected-access

    resumed = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    self.assertEqual(resumed.restore_checkpoint(), {"case": mode})
    self.assertEqual(resumed.micro_step_count, _SAVED_AFTER)
    restored_dtypes = {leaf.dtype for leaf in jax.tree.leaves(resumed._accumulated_grads)}  # pylint: disable=protected-access
    for got, want in zip(_host_leaves(resumed._accumulated_grads), saved):  # pylint: disable=protected-access
      np.testing.assert_array_equal(got, want, err_msg="the restored accumulator is not the saved one")
    # The dtype is asserted only after the step is finished, so a wrong dtype first shows as its
    # effect on the step: a TypeError from the compiled kernels, or different weights.
    resumed.compile(batches[0] if mode == "dummy" else None)
    resumed.fwd_bwd(batches[-1])
    resumed.update()

    self.assertEqual(resumed.train_step, uninterrupted.train_step)
    want, got, initial = _params(uninterrupted), _params(resumed), _params(stopped)
    self.assertEqual(got.keys(), want.keys())
    moved = max(float(np.max(np.abs(want[k] - initial[k]))) for k in want)
    self.assertGreater(moved, 0.0, "the step moved no weight, so equality below would be vacuous")
    for key in want:
      np.testing.assert_array_equal(got[key], want[key], err_msg=f"{key} differs from the uninterrupted step")
    self.assertEqual(restored_dtypes, {jnp.dtype(accumulation_dtype)}, "restored outside the accumulation dtype")


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.defaultTestLoader.loadTestsFromTestCase(IntraStepCheckpointTest)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")

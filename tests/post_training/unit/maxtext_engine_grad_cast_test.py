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

"""`cast_grads_after_all_reduce`: which gradients `fwd_bwd` returns uncast, and that the rest is unchanged.

With the key on, a gradient that `fwd_bwd` sums across devices by an all-reduce alone leaves it in
the parameters' dtype and is cast to the accumulation dtype as it joins the running sum. The rig uses
float32 weights with bfloat16 gradients, the pairing where there is a cast to move; every other test
file runs the weights and the accumulator in one dtype, where the key has nothing to do.

On CPU every cross-device sum is float32 whatever the key, so moving the cast must not change a bit
there. What it changes on TPU, the dtype of those all-reduces, is not observable on this backend.

Everything runs in a subprocess with four CPU devices, for the reason given in
`maxtext_engine_data_parallel_test.py`: a sharded mesh needs more than one device, and the device
count is fixed before this file is imported.
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
_SENTINEL = "MAXTEXT_ENGINE_GRAD_CAST_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

# The mesh shapes the tests need: parameters sharded over the batch's axis and parameters that are
# not (fsdp), every parameter replicated (pure data parallelism), the deferred all-reduce, parameters
# sharded over an axis the batch is not split over (tensor), and over the sequence's axis (context).
_FSDP = {"ici_fsdp_parallelism": _REQUIRED_DEVICES, "ici_data_parallelism": 1}
_DATA = {"ici_fsdp_parallelism": 1, "ici_data_parallelism": _REQUIRED_DEVICES}
_DEFERRED = {**_DATA, "shard_mode": "explicit"}
_TENSOR = {"ici_fsdp_parallelism": 1, "ici_data_parallelism": 2, "ici_tensor_parallelism": 2}
_CONTEXT = {
    "ici_fsdp_parallelism": 1,
    "ici_data_parallelism": 2,
    "ici_context_parallelism": 2,
    "context_parallel_load_balance": False,
}


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_grad_cast_on_a_four_device_cpu_mesh():
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
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


def _config(**overrides) -> pyconfig.HyperParameters:
  """A tiny real decoder with float32 weights and bfloat16 gradients, summed in bfloat16."""
  fields = {
      "model_name": "default",
      "run_name": "engine_grad_cast_test",
      "base_output_directory": tempfile.mkdtemp(prefix="engine_grad_cast_"),
      "enable_checkpointing": False,
      "convert_checkpoint_if_possible": False,
      "skip_jax_distributed_system": True,
      "enable_tensorboard": False,
      "record_internal_nn_metrics": False,
      "enable_dropout": False,
      "init_weights_seed": 0,
      "dtype": "float32",
      "weight_dtype": "float32",
      "grad_dtype": "bfloat16",
      "remat_policy": "none",
      "scan_layers": False,
      "attention": "dot_product",
      "ici_tensor_parallelism": 1,
      "per_device_batch_size": 1,
      "vocab_size": 128,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 2,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 4,
      "head_dim": 16,
      "max_target_length": 32,
      "opt_type": "sgd",
      "learning_rate": 0.1,
      "gradient_clipping_threshold": 0.0,
      "warmup_steps_fraction": 0.0,
      "learning_rate_final_fraction": 1.0,
      "profiler_steps": 0,
      **_FSDP,
  }
  fields.update(overrides)
  argv = ["maxtext_engine_grad_cast_test.py", get_test_config_path("base.yml")]
  return pyconfig.initialize(argv + [f"{k}={v}" for k, v in fields.items()])


def _batch(cfg: pyconfig.HyperParameters, seed: int) -> dict[str, np.ndarray]:
  """A `train.py` loss batch as host arrays, which the compiled kernel places itself."""
  rows, seq = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length
  tokens = np.random.default_rng(seed).integers(1, cfg.vocab_size, size=(rows, seq)).astype(np.int32)
  ones = np.ones((rows, seq), dtype=np.int32)
  return {
      "inputs": tokens,
      "targets": np.roll(tokens, -1, axis=-1),
      "inputs_position": np.tile(np.arange(seq, dtype=np.int32), (rows, 1)),
      "inputs_segmentation": ones,
      "targets_segmentation": ones,
  }


def _engine(**overrides) -> tuple[maxtext_engine.MaxTextTrainingEngine, pyconfig.HyperParameters]:
  """An engine compiled against a first batch, built the way Tunix builds it: no outer context."""
  cfg = _config(**overrides)
  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=maxtext_utils.get_mesh_from_config(cfg))
  engine.compile(_batch(cfg, 0))
  return engine, cfg


def _grads_out(engine) -> dict[str, jax.ShapeDtypeStruct]:
  """The gradients `fwd_bwd` returns, by parameter path, as its compiled executable declares them."""
  grads = engine._compiled_fwd_bwd.out_info[3]  # pylint: disable=protected-access
  return {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}


def _bits(tree) -> dict[str, np.ndarray]:
  """Every leaf's raw bits by path, so comparing two trees is bitwise."""
  out = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
    value = np.asarray(jax.device_get(leaf))
    out[jax.tree_util.keystr(path)] = value.view(np.uint16 if value.dtype.itemsize == 2 else np.uint32)
  return out


def _train(micro_batches: int, steps: int = 2, **overrides):
  """Trains compiled steps; returns the running sum before each update and the final weights, as bits."""
  engine, cfg = _engine(**overrides)
  sums = []
  for step in range(steps):
    for micro_batch in range(micro_batches):
      engine.fwd_bwd(_batch(cfg, seed=step * micro_batches + micro_batch))
    sums.append(_bits(engine._reduced_accumulated_grads()))  # pylint: disable=protected-access
    engine.update()
  weights = _bits(nnx.state(engine.state.model, nnx.Param))
  engine.close()
  return sums, weights


class GradCastTest(parameterized.TestCase):
  """The key on the tiny decoder, against the same decoder with the key off."""

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def _assert_bitwise_equal(self, got: dict, want: dict, what: str):
    self.assertEqual(set(got), set(want), what)
    for key, value in want.items():
      np.testing.assert_array_equal(got[key], value, err_msg=f"{what}: {key}")

  def test_fwd_bwd_returns_the_all_reduced_gradients_uncast(self):
    """Under fsdp the replicated parameters (the norm scales) are all-reduced and leave uncast; the rest are cast."""
    engine, _ = _engine(cast_grads_after_all_reduce=True)
    params = nnx.state(engine.state.model, nnx.Param)
    replicated = {
        jax.tree_util.keystr(path): leaf.sharding.is_fully_replicated
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
    }
    grads = _grads_out(engine)
    engine.close()

    self.assertEqual(set(grads), set(replicated))
    for path, grad in grads.items():
      want = jnp.float32 if replicated[path] else jnp.bfloat16
      self.assertEqual(grad.dtype, want, f"{path} {grad.shape}")
    uncast = sorted(path for path, is_replicated in replicated.items() if is_replicated)
    self.assertTrue(uncast and all("norm" in path for path in uncast), uncast)
    self.assertLess(len(uncast), len(grads), "no gradient was sharded, so the cast-inside branch never ran")

  def test_key_off_casts_every_gradient_inside_fwd_bwd(self):
    engine, _ = _engine()
    grads = _grads_out(engine)
    engine.close()
    self.assertEqual({str(grad.dtype) for grad in grads.values()}, {"bfloat16"})

  def test_selection_follows_sharding_not_size(self):
    """Pure data parallelism replicates every parameter, so every gradient is all-reduced and leaves uncast.

    That includes the embedding, which is 2^20 elements here: the selection does not depend on size.
    """
    engine, cfg = _engine(cast_grads_after_all_reduce=True, vocab_size=16384, **_DATA)
    grads = _grads_out(engine)
    engine.close()
    embedding = [grad for path, grad in grads.items() if "token_embedder" in path]
    self.assertLen(embedding, 1)
    self.assertEqual(int(np.prod(embedding[0].shape)), cfg.vocab_size * cfg.emb_dim)
    self.assertEqual(int(np.prod(embedding[0].shape)), 1 << 20)
    self.assertEqual({str(grad.dtype) for grad in grads.values()}, {"float32"})

  def test_deferred_all_reduce_leaves_nothing_uncast(self):
    """Under the deferred all-reduce `fwd_bwd` reduces no gradient, so the key casts every one inside it."""
    engine, _ = _engine(cast_grads_after_all_reduce=True, **_DEFERRED)
    deferred = engine._unreduced_grad_shardings is not None  # pylint: disable=protected-access
    grads = _grads_out(engine)
    engine.close()
    self.assertTrue(deferred, "the deferral did not engage, so this test checks nothing")
    self.assertEqual({str(grad.dtype) for grad in grads.values()}, {"bfloat16"})

  def test_selection_follows_the_batch_axes_not_replication(self):
    """Under data x tensor parallelism a parameter sharded over `tensor` is all-reduced over `data` alone.

    So every gradient leaves uncast, the sharded ones as well as the replicated ones.
    """
    engine, _ = _engine(cast_grads_after_all_reduce=True, **_TENSOR)
    params = nnx.state(engine.state.model, nnx.Param)
    sharded = [leaf for leaf in jax.tree.leaves(params) if not leaf.sharding.is_fully_replicated]
    grads = _grads_out(engine)
    engine.close()
    self.assertTrue(sharded, "no parameter is sharded over tensor, so this test checks nothing")
    self.assertEqual({str(grad.dtype) for grad in grads.values()}, {"float32"})

  def test_sequence_axis_counts_as_a_batch_axis(self):
    """Under context parallelism a parameter sharded over `context` is cast inside; the replicated ones leave uncast."""
    engine, _ = _engine(cast_grads_after_all_reduce=True, **_CONTEXT)
    params = nnx.state(engine.state.model, nnx.Param)
    replicated = {
        jax.tree_util.keystr(path): leaf.sharding.is_fully_replicated
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
    }
    grads = _grads_out(engine)
    engine.close()
    self.assertFalse(all(replicated.values()), "no parameter is sharded over context, so this test checks nothing")
    for path, grad in grads.items():
      self.assertEqual(grad.dtype, jnp.float32 if replicated[path] else jnp.bfloat16, f"{path} {grad.shape}")

  def test_nothing_is_selected_without_a_cast_to_move(self):
    """With float32 weights and float32 gradients no gradient is cast, so no leaf is selected and nothing is copied."""
    engine, _ = _engine(cast_grads_after_all_reduce=True, grad_dtype="float32", **_DATA)
    mask, cast = engine._uncast_grad_mask, engine._cast_uncast_grads  # pylint: disable=protected-access
    engine.close()
    self.assertIsNone(mask)
    self.assertIsNone(cast)

  def test_running_sum_stays_in_the_accumulation_dtype(self):
    """The uncast gradients are cast when they start the running sum and whenever they join it."""
    engine, cfg = _engine(cast_grads_after_all_reduce=True)
    for micro_batch in range(3):
      engine.fwd_bwd(_batch(cfg, micro_batch))
      dtypes = {str(leaf.dtype) for leaf in jax.tree.leaves(engine._accumulated_grads)}  # pylint: disable=protected-access
      self.assertEqual(dtypes, {"bfloat16"}, f"running sum after micro-batch {micro_batch + 1}")
    engine.update()
    engine.close()

  @parameterized.named_parameters(("one_micro_batch", 1), ("three_micro_batches", 3))
  def test_moving_the_cast_changes_no_number_on_cpu(self, micro_batches):
    """The running sums and the trained weights are bitwise those of the key off."""
    sums, weights = _train(micro_batches, cast_grads_after_all_reduce=True)
    want_sums, want_weights = _train(micro_batches)
    for step, (got, want) in enumerate(zip(sums, want_sums)):
      self._assert_bitwise_equal(got, want, f"running sum before update {step}")
    self._assert_bitwise_equal(weights, want_weights, "trained weights")


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  _result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GradCastTest))
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")

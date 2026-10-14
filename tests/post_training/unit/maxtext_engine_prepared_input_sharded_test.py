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

"""prepare_fwd_bwd_input with Tunix's real model input, on a four-device mesh.

`maxtext_engine_prepared_input_test` uses a one-device mesh and a flat toy input. Production feeds Tunix's
`_algo_model_input`: {"train_example": RLTrainerPayload (a flax.struct pytree with static `metadata`),
"algo_config": a closed-over object, "pad_id", "eos_id"}, on a mesh whose batch axis shards. Here the batch is split
over four host devices, and a prepared micro-batch, with `wait_for_device` after it as Tunix's
`TrainerWorker.stage_fwd_bwd` does, must train bit-identically to the payload it came from.

Everything runs in a subprocess, for the reason `maxtext_engine_data_parallel_test.py` gives: the CPU backend reads
`--xla_force_host_platform_device_count` only when it initializes, which has already happened by the time pytest
imports this file, so setting the flag here would leave one device and skip the tests green.
`test_prepared_input_on_a_four_device_cpu_mesh` re-execs the module with the flag appended and refuses to pass unless
the child reports tests actually run.
"""
# pylint: disable=protected-access

import dataclasses
import functools
import os
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
import numpy as np
import optax
import pytest

from tests.utils.test_helpers import get_test_config_path

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]

_REQUIRED_DEVICES = 4
_SENTINEL = "MAXTEXT_ENGINE_PREPARED_INPUT_SHARDED_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")

B, P, C, L, K = 4, 8, 16, 2, 3


@pytest.mark.post_training
@pytest.mark.cpu_only
def test_prepared_input_on_a_four_device_cpu_mesh():
  """The only test pytest collects here; the class below runs inside the child process."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}".strip()
  env["JAX_PLATFORMS"] = "cpu"
  # The child imports `tests.utils.test_helpers`, which pytest puts on the path for us and a bare interpreter does
  # not.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root

  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)

  report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert result.returncode == 0, report
  ran = _RAN.search(result.stdout)
  # An exit status of 0 is also what a run that skipped everything produces.
  assert ran, f"the child did not report a completed run\n{report}"
  assert int(ran.group(1)) > 0, f"every test in the child skipped\n{report}"


class _Model(nnx.Module):

  def __init__(self):
    self.weights = nnx.Param(jnp.array([1.0, 2.0, 0.5]))


def _payload(seed: int):
  from tunix.experimental.common import datatypes  # pylint: disable=import-outside-toplevel

  rng = np.random.default_rng(seed)
  return datatypes.RLTrainerPayload(
      prompt_ids=rng.integers(1, 50, (B, P)).astype(np.int32),
      prompt_mask=np.ones((B, P), np.float32),
      completion_ids=rng.integers(1, 50, (B, C)).astype(np.int32),
      completion_mask=(rng.random((B, C)) > 0.2).astype(np.float32),
      advantages=rng.normal(size=(B,)).astype(np.float32),
      old_per_token_logps=-rng.random((B, C)).astype(np.float32),
      routed_experts=rng.integers(0, 8, (B, P + C, L, K)).astype(np.int16),
      metadata={"trajectory_ids": tuple(f"t{seed}_{i}" for i in range(B))},
  )


def _loss_fn(model, train_example, algo_config, pad_id, eos_id):
  del eos_id
  w = model.weights[...]
  x = train_example.completion_ids.astype(jnp.float32) / 50.0
  m = train_example.completion_mask * (train_example.completion_ids != pad_id)
  routed = jnp.mean(train_example.routed_experts.astype(jnp.float32), (2, 3))
  pred = w[0] * x + w[1] * x**2 * algo_config.scale + w[2] * routed[:, P:] / 8.0 + train_example.old_per_token_logps
  per_tok = pred * train_example.advantages[:, None] * m
  return (
      abstract_engine.WeightedMetric(unreduced_sum=jnp.sum(per_tok), denominator=jnp.sum(m)),
      {},
  )


class NestedPayloadPreparedInputTest(absltest.TestCase):

  __test__ = False  # collected only via the subprocess entry point at the top of this file.

  def setUp(self):
    """A four-device mesh and a toy three-weight model, built without a checkpoint."""
    super().setUp()
    if jax.device_count() < _REQUIRED_DEVICES:
      self.skipTest(f"needs {_REQUIRED_DEVICES} devices; run through pytest, which re-execs with them")
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        model_name="llama3.1-8b",
        run_name="test_run",
        base_output_directory=self.create_tempdir().full_path,
        init_weights_seed=42,
        micro_batch_size_to_train_on=B,
        gradient_accumulation_steps=1,
        enable_dropout=False,
        record_internal_nn_metrics=False,
        enable_tensorboard=False,
        tensorboard_dir=self.create_tempdir().full_path,
        skip_jax_distributed_system=True,
        enable_checkpointing=False,
        profiler_steps=0,
    )
    self.mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)
    for patcher in (
        mock.patch.object(
            maxtext_engine.train_utils,
            "create_training_optimizer",
            return_value=(lambda step: jnp.array(0.001), optax.adam(0.05)),
        ),
        mock.patch.object(
            maxtext_engine.model_creation_utils,
            "from_pretrained",
            side_effect=lambda *a, **k: (_Model(), self.mesh),
        ),
    ):
      self.addCleanup(patcher.stop)
      patcher.start()
    self.algo_config = types.SimpleNamespace(scale=0.75)

  def _engine(self):
    """An engine fed through Tunix's `_algo_model_input`, compiled on the first payload."""
    from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=import-outside-toplevel

    engine = maxtext_engine.MaxTextTrainingEngine(self.config)
    engine.with_loss_fn(_loss_fn, has_aux=True)
    engine.with_gen_model_input_fn(
        functools.partial(
            algorithm_adapter._algo_model_input,
            algo_config=self.algo_config,
            pad_id=0,
            eos_id=1,
        )
    )
    engine.compile(_payload(0))
    return engine

  def _train(self, engine, payloads, prepare_at):
    """Two steps of two micro-batches; those in `prepare_at` prepared ahead, after the previous update."""
    prepared = {}
    if 0 in prepare_at:
      prepared[0] = engine.prepare_fwd_bwd_input(payloads[0])
      engine.wait_for_device()
    metrics = []
    for step in range(2):
      for j in range(2):
        i = 2 * step + j
        engine.fwd_bwd(prepared.pop(i) if i in prepared else payloads[i])
      engine.update()
      if 2 * step + 2 in prepare_at:
        # Prepared while the update is still on the device, as stage_fwd_bwd does, which then waits for the device
        # to report whether that hid.
        prepared[2 * step + 2] = engine.prepare_fwd_bwd_input(payloads[2 * step + 2])
        engine.wait_for_device()
      buf = engine.get_metrics(clear_cache=True)
      metrics.append(np.asarray(buf.weighted_metrics["loss"].compute()).tolist())
    return np.asarray(engine.model.weights[...]), metrics

  def test_bit_identical_on_a_sharded_mesh(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    ref, new = self._engine(), self._engine()
    with (
        mock.patch.object(ref, "_compile_for_batch", wraps=ref._compile_for_batch) as ref_compiles,
        mock.patch.object(new, "_compile_for_batch", wraps=new._compile_for_batch) as new_compiles,
    ):
      ref_w, ref_m = self._train(ref, payloads, prepare_at=set())
      new_w, new_m = self._train(new, payloads, prepare_at={0, 2})
    np.testing.assert_array_equal(ref_w, new_w)
    self.assertEqual(ref_m, new_m)
    self.assertEqual(ref_compiles.call_count, new_compiles.call_count)

  def test_prepared_leaves_are_sharded_like_the_compile(self):
    engine = self._engine()
    prepared = engine.prepare_fwd_bwd_input(_payload(2))
    dynamic, static = maxtext_engine._split_static_and_dynamic(prepared.batch)
    self.assertEqual(set(dynamic), {"train_example"})
    self.assertEqual(set(static), {"algo_config", "pad_id", "eos_id"})
    shardings = engine._batch_data_shardings(dynamic)
    example, sharding = dynamic["train_example"], shardings["train_example"]
    self.assertEqual(example.metadata, {})  # stripped by _prepare_batch
    split_2d = 0
    for field in (
        "completion_ids",
        "completion_mask",
        "old_per_token_logps",
        "routed_experts",
        "advantages",
        "prompt_ids",
    ):
      leaf, want = getattr(example, field), getattr(sharding, field)
      self.assertIsInstance(leaf, jax.Array, field)
      self.assertEqual(leaf.sharding, want, field)
      rows = {s.index[0] for s in leaf.addressable_shards}
      if leaf.ndim >= 2 and len(rows) == _REQUIRED_DEVICES:
        split_2d += 1
    # Not vacuous: the 2-D leaves really are split four ways.
    self.assertGreaterEqual(split_2d, 3)

  def test_one_token_change_is_seen(self):
    payloads = [_payload(s) for s in (1, 2, 3, 4)]
    changed = list(payloads)
    ids = np.array(payloads[2].completion_ids)
    ids[0, 0] = ids[0, 0] % 49 + 1
    changed[2] = dataclasses.replace(payloads[2], completion_ids=ids)
    ref_w, _ = self._train(self._engine(), payloads, prepare_at=set())
    new_w, _ = self._train(self._engine(), changed, prepare_at={2})
    self.assertFalse(np.array_equal(ref_w, new_w))


_SUITE = (NestedPayloadPreparedInputTest,)


if __name__ == "__main__":
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(
        f"needs {_REQUIRED_DEVICES} devices, got {jax.device_count()}; run this through pytest, which sets "
        f"XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
    )
  flags.FLAGS(sys.argv[:1])  # absltest's create_tempdir reads --test_tmpdir; a bare unittest run parses no flags.
  _loader = unittest.defaultTestLoader
  _result = unittest.TextTestRunner(verbosity=2).run(
      unittest.TestSuite(_loader.loadTestsFromTestCase(cls) for cls in _SUITE)
  )
  if not _result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={_result.testsRun - len(_result.skipped)}")

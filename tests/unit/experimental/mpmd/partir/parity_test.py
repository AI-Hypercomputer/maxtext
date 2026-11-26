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

"""Tests for the MPMD-vs-SPMD parity check's comparison and gating (no MPMD run, no Shardy)."""

# pylint: disable=protected-access

import dataclasses
import math
import types
import unittest
from unittest import mock

import jax
import numpy as np
import optax

from maxtext.common.common_types import DecoderBlockType
from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import parity
from maxtext.experimental.mpmd.partir import train_step

TOLERANCES = (1e-4, 1e-2, 1e-2)  # loss, param, grad: parity's flag defaults


def tree_diff(errors, norm_floor=1e-6):
  """A TreeDiff over tensors 'router' and 'dense' that differ by the given relative amounts."""
  reference = {"router": np.ones((4, 4)), "dense": np.arange(1.0, 17.0).reshape(4, 4)}
  run = {name: value * (1 + errors.get(name, 0.0)) for name, value in reference.items()}
  return parity._compare_params(run, reference, norm_floor=norm_floor)


def comparison(losses, reference, grads=None, params=None):
  return parity.Comparison(
      losses=tuple(losses),
      reference_losses=tuple(reference),
      grads=tree_diff(grads or {}),
      params=tree_diff(params or {}, norm_floor=1.0),
  )


def result(mpmd, floor=None):
  return parity.ParityResult(mpmd=mpmd, floor=floor, floor_microbatches=4 if floor else 0, num_params=2)


class GateTest(unittest.TestCase):

  def test_floor_microbatches(self):
    self.assertEqual([parity.noise_floor_microbatches(n, -1) for n in (1, 2, 4, 5, 8, 9)], [0, 1, 2, 1, 4, 3])
    self.assertEqual(parity.noise_floor_microbatches(8, 0), 0)
    self.assertEqual(parity.noise_floor_microbatches(8, 2), 2)

  def test_fixed_tolerances_without_a_floor(self):
    self.assertTrue(
        result(comparison([2.0, 1.0], [2.0, 1.00005], {"dense": 1e-3}, {"dense": 1e-3})).passed(*TOLERANCES, 4.0)
    )
    bad = result(comparison([2.0, 1.0], [2.0, 1.001], {"dense": 1e-3}, {"dense": 1e-3}))
    self.assertFalse(bad.passed(*TOLERANCES, 4.0))
    self.assertIn("bound 1.000e-04 = the tolerance", bad.report(*TOLERANCES, 4.0))

  def test_floor_widens_each_bound_and_only_its_own(self):
    """MoE on TPU: MPMD as far from the reference as the floor run is passes on the floor."""
    floor = comparison(
        [12.36, 10.985], [12.36, 10.99], {"router": 0.035, "dense": 0.005}, {"router": 0.015, "dense": 0.001}
    )
    mpmd = comparison([12.36, 10.98], [12.36, 10.99], {"router": 0.04, "dense": 0.006}, {"router": 0.017, "dense": 0.002})
    self.assertTrue(result(mpmd, floor).passed(*TOLERANCES, 4.0))
    self.assertFalse(result(mpmd).passed(*TOLERANCES, 4.0))
    self.assertIn("4 x floor", result(mpmd, floor).report(*TOLERANCES, 4.0))

  def test_grad_gate_on_its_own(self):
    """Loss and parameters sit at their floor; only the gradients exceed theirs at factor 1."""
    floor = comparison([12.36, 10.985], [12.36, 10.99], {"router": 0.035}, {"router": 0.015})
    mpmd = comparison([12.36, 10.985], [12.36, 10.99], {"router": 0.04}, {"router": 0.015})
    loss, grads, params = result(mpmd, floor).gates(*TOLERANCES, 1.0)
    self.assertEqual((loss.passed, grads.passed, params.passed), (True, False, True))
    self.assertEqual(grads.where, "['router']")
    # Parameter differences far above the parameter floor fail even when the gradient floor is wide.
    lopsided = comparison([12.36, 10.985], [12.36, 10.99], {"router": 0.04}, {"router": 0.05})
    self.assertFalse(result(lopsided, floor).passed(*TOLERANCES, 3.0))

  def test_each_tensor_against_its_own_floor(self):
    """A noisy router floor (3.5%) must not license a 12% error on another tensor (floor 0.5%)."""
    floor = comparison([1.0], [1.0], {"router": 0.035, "dense": 0.005})
    mpmd = comparison([1.0], [1.0], {"router": 0.03, "dense": 0.12})
    _, grads, _ = result(mpmd, floor).gates(*TOLERANCES, 4.0)
    self.assertFalse(grads.passed)
    self.assertEqual((grads.where, round(grads.bound, 6)), ("['dense']", 0.02))

  def test_step1_loss_against_the_floor_up_to_step1(self):
    """A chaotic late step of the floor (2e-4) must not loosen step 1 (floor 1e-6)."""
    floor = comparison([1.000001, 1.00001, 1.0002], [1.0, 1.0, 1.0])
    mpmd = parity.Comparison(
        losses=(1.0005, 1.0, 1.0), reference_losses=(1.0, 1.0, 1.0), grads=floor.grads, params=floor.params
    )
    loss, _, _ = result(mpmd, floor).gates(*TOLERANCES, 4.0)
    self.assertFalse(loss.passed)
    self.assertEqual(loss.where, "step 1")
    late = parity.Comparison(
        losses=(1.0, 1.0, 1.0005), reference_losses=(1.0, 1.0, 1.0), grads=floor.grads, params=floor.params
    )
    self.assertTrue(result(late, floor).passed(*TOLERANCES, 4.0))  # 5e-4 <= 4 x 2e-4

  def test_running_maximum_of_the_floor(self):
    """Step 2 is bounded by the floor's largest difference up to step 2, not by its own smaller one."""
    floor = comparison([1.0003, 1.00001], [1.0, 1.0])  # floor 3e-4 at step 1, 1e-5 at step 2
    mpmd = comparison([1.0, 1.001], [1.0, 1.0])  # 1e-3 at step 2 <= 4 x 3e-4
    loss, _, _ = result(mpmd, floor).gates(*TOLERANCES, 4.0)
    self.assertTrue(loss.passed)
    self.assertAlmostEqual(loss.floor, 3e-4, places=9)

  def test_losses_below_1_compared_absolutely(self):
    """A 1.3e-5 difference at a memorized loss of 0.031 once read 4.3e-4 and failed (deepseek2-16b cut to 2 layers)."""
    reference = [10.868768692, 2.304498196, 0.030810289]
    floor = comparison([10.868770600, 2.304684, 0.030811244], reference)
    mpmd = comparison([10.868772507, 2.304545879, 0.030823506], reference)
    self.assertAlmostEqual(mpmd.loss_diffs[2], 1.3217e-05, places=8)
    self.assertTrue(result(mpmd, floor).passed(*TOLERANCES, 4.0))
    report = result(mpmd, floor).report(*TOLERANCES, 4.0)  # its columns once read 'rel diff'
    self.assertIn("|MPMD - SPMD| / max(|SPMD|, 1)", report)
    self.assertNotIn("rel diff", report)
    self.assertAlmostEqual(parity.relative_difference(10.001, 10.0), 1e-4, places=12)  # relative above 1
    self.assertFalse(result(comparison([2.0, 0.5], [2.0, 0.4998])).passed(*TOLERANCES, 4.0))  # 2e-4

  def test_non_finite_values_fail_and_a_non_finite_floor_is_ignored(self):
    good = comparison([2.0, 1.0], [2.0, 1.0])
    for mpmd in (
        comparison([2.0, math.nan], [2.0, 1.0]),  # MPMD loss NaN
        comparison([2.0, 1.0], [2.0, math.inf]),  # reference loss inf
        comparison([2.0, 1.0], [2.0, 1.0], {"dense": math.nan}),  # NaN gradients
        comparison([2.0, 1.0], [2.0, 1.0], params={"router": math.inf}),
    ):
      self.assertFalse(result(mpmd, good).passed(*TOLERANCES, 4.0))
      self.assertFalse(result(mpmd).passed(*TOLERANCES, 4.0))
    diverged_floor = comparison([2.0, math.inf], [2.0, 1.0], {"dense": math.nan})
    five_percent = comparison([2.0, 1.05], [2.0, 1.0], {"dense": 0.5}, {"dense": 0.5})
    self.assertIsNone(result(five_percent, diverged_floor).usable_floor)
    self.assertFalse(result(five_percent, diverged_floor).passed(*TOLERANCES, 4.0))
    self.assertIn("ignored", result(five_percent, diverged_floor).report(*TOLERANCES, 4.0))
    self.assertEqual(tree_diff({"dense": math.nan}).rel_l2, math.inf)

  def test_assemble_result_from_the_three_runs(self):
    """The comparison of the three runs on host trees (CheckParityTest runs check_parity itself)."""
    reference = {"router": np.ones((4, 4)), "dense": np.arange(1.0, 17.0).reshape(4, 4)}
    nudged = {name: value * 1.001 for name, value in reference.items()}
    outcome = parity.assemble_result(
        reference=([2.0, 1.0], reference, reference),
        mpmd_run=([2.0, 1.00005], nudged, nudged),
        floor=([2.0, 1.00004], reference, reference),
        floor_microbatches=4,
        notes=["a note"],
    )
    self.assertEqual((outcome.num_params, outcome.floor_microbatches, outcome.notes), (2, 4, ("a note",)))
    self.assertAlmostEqual(outcome.mpmd.grads.rel_l2, 1e-3, places=6)
    self.assertTrue(outcome.passed(*TOLERANCES, 4.0))
    self.assertIn("notes:", outcome.report(*TOLERANCES, 4.0))
    alone = parity.assemble_result(reference=([2.0], reference, reference), mpmd_run=([2.0], nudged, nudged))
    self.assertIsNone(alone.floor)

  def test_gradients_far_below_1_against_their_own_norm(self):
    """Step-1 gradient norms are far below 1 (deepseek3-tiny's router about 2e-3 on CPU): assemble_result floors the
    gradients' denominator at 1e-5, not at the parameters' 1.0, under which a 50% error on such a tensor read 3.2e-4."""
    params = {"w": np.ones((4, 4))}
    reference = {"router": np.full((64, 64), 1e-5), "tiny": np.full((4, 4), 1e-7)}  # norms 6.4e-4 and 4e-7
    off = {name: grad * 1.5 for name, grad in reference.items()}
    outcome = parity.assemble_result(reference=([2.0], reference, params), mpmd_run=([2.0], off, params))
    loss, grads, params_gate = outcome.gates(*TOLERANCES, 4.0)
    self.assertEqual((loss.passed, grads.passed, params_gate.passed), (True, False, True))
    self.assertFalse(outcome.passed(*TOLERANCES, 4.0))
    rel = {path: value for value, path, _ in outcome.mpmd.grads.leaves}
    self.assertAlmostEqual(rel["['router']"], 0.5, places=9)
    self.assertAlmostEqual(rel["['tiny']"], 2e-7 / 1e-5, places=9)  # a norm below the floor: divided by 1e-5

  def test_notes(self):
    def config(**kw):
      return types.SimpleNamespace(
          **{
              "dtype": "bfloat16",
              "num_experts": 1,
              "num_experts_per_tok": 1,
              "matmul_precision": "default",
              "decoder_block": DecoderBlockType.DEEPSEEK,
              **kw,
          }
      )

    self.assertEqual(parity.parity_notes(config(dtype="float32"), 2), [])
    notes = parity.parity_notes(config(), 2)
    self.assertEqual(len(notes), 1)
    bf16 = notes[0]
    self.assertIn("bfloat16", bf16)
    self.assertEqual(parity.parity_notes(config(), 1), [])  # no data parallelism
    notes = parity.parity_notes(config(dtype="float32", num_experts=16, num_experts_per_tok=8), 1)
    self.assertEqual(len(notes), 1)
    moe = notes[0]
    self.assertIn("--maxtext_overrides=num_experts_per_tok=16", moe)
    self.assertEqual(parity.parity_notes(config(dtype="float32", num_experts=16, num_experts_per_tok=16), 2), [])
    # On TPU a float32 matmul at the default precision runs in bfloat16 passes.
    notes = parity.parity_notes(config(dtype="float32"), 2, platform="tpu")
    self.assertEqual(len(notes), 1)
    self.assertIn("--maxtext_overrides=matmul_precision=highest", notes[0])
    self.assertEqual(parity.parity_notes(config(dtype="float32", matmul_precision="highest"), 2, platform="tpu"), [])
    # matmul_precision reaches only the simple model's output head; its layers take JAX's default precision.
    simple = config(dtype="float32", matmul_precision="highest", decoder_block=DecoderBlockType.SIMPLE_MLP)
    notes = parity.parity_notes(simple, 2, platform="tpu")
    self.assertEqual(len(notes), 1)
    self.assertIn("add JAX_DEFAULT_MATMUL_PRECISION=highest", notes[0])
    self.assertNotIn("matmul_precision=highest and", notes[0])
    with jax.default_matmul_precision("highest"):
      self.assertEqual(parity.parity_notes(simple, 2, platform="tpu"), [])
      self.assertIn("matmul_precision=highest.", parity.parity_notes(config(dtype="float32"), 2, platform="tpu")[0])

  def test_step1_gradients_from_the_first_moment(self):
    params = {"w": np.ones((2,), np.float32)}
    tx = optax.adamw(1e-3)
    _, state = tx.update({"w": np.full((2,), 3.0, np.float32)}, tx.init(params), params)
    np.testing.assert_allclose(parity._first_moment(state)["w"], [0.3, 0.3], rtol=1e-6)
    np.testing.assert_allclose(parity._step1_gradients(state)["w"], [3.0, 3.0], rtol=1e-6)


class CheckParityTest(unittest.TestCase):

  def test_check_parity_end_to_end_on_cpu(self):
    """check_parity's whole path on CPU, with the Pathways-only MPMD compile swapped for the SPMD step.

    Both arms then run the same program, so they agree exactly; what this checks is the code around
    them (a renamed variable there once crashed every parity run after its compute).
    """
    if len(jax.devices("cpu")) < 4:
      self.skipTest("needs 4 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    real_build = parity.runner_lib.build

    def build(hparams, *, mode, **kw):
      del mode  # the "MPMD" arm runs the SPMD step too
      return real_build(hparams, mode=train_step.ExecutionMode.SPMD, **kw)

    class Lowered:

      def __init__(self, fn):
        self.fn, self.function_named_shardings = fn, types.SimpleNamespace(input_specs=None)

      def compile(self):
        return jax.jit(self.fn)

    hparams = hp.Hparams(
        model_name="simple",
        num_layers=2,
        num_stages=2,
        emb_dim=64,
        mlp_dim=128,
        vocab_size=256,
        max_target_length=16,
        per_device_batch_size=2,
        num_microbatches=2,
        steps=2,
    )
    with (
        mock.patch.object(parity.runner_lib, "build", build),
        mock.patch.object(parity.runner_lib.Runnable, "lower", lambda runnable: Lowered(runnable.train_step)),
        mock.patch.object(parity.runner_lib, "place_on_stage_meshes", lambda inputs, specs: inputs),
    ):
      outcome = parity.check_parity(hparams, floor_microbatches=1)
    self.assertEqual(outcome.mpmd.max_rel_loss_diff, 0.0)
    self.assertEqual((outcome.floor_microbatches, len(outcome.mpmd.losses)), (1, 2))
    self.assertTrue(outcome.passed(*TOLERANCES, 4.0))
    with self.assertRaisesRegex(ValueError, "at least 1"):
      parity.check_parity(dataclasses.replace(hparams, steps=0))

  def test_the_reference_compiles_once(self):
    """Pinned to its device: DeepSeek's layers once returned step 1's state committed to another sharding, and the
    step compiled again for step 2 (3.5 min for deepseek3-tiny with mHC on TPU7x)."""
    if len(jax.devices("cpu")) < 2:
      self.skipTest("needs 2 CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    hparams = hp.Hparams(
        model_name="deepseek3-tiny",
        num_layers=2,
        first_num_dense_layers=1,
        num_stages=2,
        emb_dim=64,
        mlp_dim=128,
        vocab_size=128,
        max_target_length=16,
        per_device_batch_size=2,
        num_microbatches=2,
    )
    runnable = parity.runner_lib.build(hparams, mode=train_step.ExecutionMode.SPMD)
    real_jit, steps = jax.jit, []

    def jit(fun, *args, **kwargs):
      jitted = real_jit(fun, *args, **kwargs)
      if fun is runnable.train_step:
        steps.append(jitted)
      return jitted

    with mock.patch.object(parity.jax, "jit", jit):
      losses, _, _ = parity._run_reference(runnable, 3)
    self.assertEqual(len(losses), 3)
    self.assertEqual([step._cache_size() for step in steps], [1])


if __name__ == "__main__":
  unittest.main()

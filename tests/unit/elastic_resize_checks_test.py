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

"""Tests for `elastic_resize_checks`."""

import json
import math
import os

from absl.testing import absltest
from absl.testing import flagsaver
from tests.utils import elastic_resize_checks as checks


def _write(path, rows):
  """Writes `rows` as a MaxText metrics file (one JSON object per line)."""
  with open(path, "wt", encoding="utf8") as metrics_file:
    for row in rows:
      metrics_file.write(json.dumps(row) + "\n")


def _rows(losses, grad_norms=None, total_weights=1024, start_step=0):
  """Per-step metric rows with consecutive steps from `start_step`."""
  grad_norms = grad_norms or [1.0] * len(losses)
  return [
      {
          "step": start_step + offset,
          "learning/loss": loss,
          "learning/grad_norm": grad_norm,
          "learning/total_weights": total_weights,
      }
      for offset, (loss, grad_norm) in enumerate(zip(losses, grad_norms, strict=True))
  ]


class ElasticResizeChecksTest(absltest.TestCase):
  """The metrics-file checks of `elastic_resize_checks`."""

  def setUp(self):
    super().setUp()
    self.tmp = self.create_tempdir().full_path

  def _main(self, reference, candidate):
    """Runs the checker's entry point with `--reference` and `--candidate`, restoring the flags afterwards.

    Only the checker's own flags are set (`flagsaver.as_parsed`): parsing the global `FLAGS` would also run the
    validators of flags that other test modules define in the same process, such as a required flag.
    """
    with flagsaver.as_parsed(reference=[reference], candidate=candidate):
      return checks.main(["elastic_resize_checks"])

  def _path(self, name, rows):
    path = os.path.join(self.tmp, name)
    _write(path, rows)
    return path

  def test_read_metrics_keeps_the_later_entry_of_a_rewound_step(self):
    path = self._path("rewound.txt", _rows([3.0, 2.9, 2.8]) + _rows([2.85, 2.75], start_step=1))
    metrics = checks.read_metrics(path)
    self.assertEqual(sorted(metrics), [0, 1, 2])
    self.assertEqual(metrics[1]["learning/loss"], 2.85)

  def test_total_weights_must_match_exactly(self):
    reference = checks.read_metrics(self._path("reference.txt", _rows([3.0, 2.9], total_weights=1024)))
    candidate = checks.read_metrics(self._path("candidate.txt", _rows([3.0, 2.9], total_weights=1056)))
    self.assertEqual(checks.check_total_weights(reference, candidate), [0, 1])
    same = checks.read_metrics(self._path("same.txt", _rows([3.1, 2.8], total_weights=1024)))
    self.assertEqual(checks.check_total_weights(reference, same), [])

  def test_envelope_uses_the_seed_to_seed_noise_floor(self):
    reference_a = checks.read_metrics(self._path("a.txt", _rows([3.0, 2.9, 2.8])))
    reference_b = checks.read_metrics(self._path("b.txt", _rows([3.01, 2.92, 2.8])))  # floor = 0.02
    inside = checks.read_metrics(self._path("in.txt", _rows([3.03, 2.9, 2.78])))  # max diff 0.03 <= 2 * 0.02
    outside = checks.read_metrics(self._path("out.txt", _rows([3.05, 2.9, 2.8])))  # 0.05 > 0.04
    references = [reference_a, reference_b]
    self.assertAlmostEqual(checks.noise_floor("learning/loss", references), 0.02)
    self.assertTrue(checks.check_envelope("learning/loss", references, inside).passed)
    self.assertFalse(checks.check_envelope("learning/loss", references, outside).passed)

  def test_single_reference_falls_back_to_absolute_tolerance(self):
    reference = checks.read_metrics(self._path("reference.txt", _rows([3.0, 2.9])))
    candidate = checks.read_metrics(self._path("candidate.txt", _rows([3.04, 2.9])))
    self.assertIsNone(checks.noise_floor("learning/loss", [reference]))
    self.assertTrue(checks.check_envelope("learning/loss", [reference], candidate, absolute_tolerance=0.05).passed)
    self.assertFalse(checks.check_envelope("learning/loss", [reference], candidate, absolute_tolerance=0.01).passed)

  def test_run_checks_and_main(self):
    reference = self._path("reference.txt", _rows([3.0, 2.9, 2.8], [1.0, 0.9, 0.8]))
    good = self._path("good.txt", _rows([3.001, 2.9, 2.8], [1.0, 0.9, 0.8]))
    bad_weights = self._path("bad_weights.txt", _rows([3.0, 2.9, 2.8], total_weights=768))
    bad_loss = self._path("bad_loss.txt", _rows([3.5, 2.9, 2.8]))
    disjoint = self._path("disjoint.txt", _rows([3.0], start_step=100))
    self.assertEqual(checks.run_checks([reference], good), [])
    self.assertIn("learning/total_weights", checks.run_checks([reference], bad_weights)[0])
    self.assertIn("learning/loss exceeds", checks.run_checks([reference], bad_loss)[0])
    self.assertIn("share no training steps", checks.run_checks([reference], disjoint)[0])
    self.assertEqual(self._main(reference, good), 0)
    self.assertEqual(self._main(reference, bad_loss), 1)

  def test_read_metrics_keeps_only_training_rows(self):
    eval_row = {"step": 1, "eval/avg_loss": 3.3, "eval/total_weights": 512}
    path = self._path("with_eval.txt", _rows([3.0, 2.9]) + [eval_row])
    metrics = checks.read_metrics(path)
    self.assertEqual(sorted(metrics), [0, 1])
    self.assertEqual(metrics[1]["learning/loss"], 2.9)
    self.assertEqual(checks.eval_row_steps(path), [1])

  def test_eval_rows_fail_the_checks(self):
    reference = self._path("reference.txt", _rows([3.0, 2.9]))
    candidate = self._path("candidate.txt", _rows([3.0, 2.9]) + [{"step": 0, "eval/avg_loss": 3.3}])
    failures = checks.run_checks([reference], candidate)
    self.assertLen(failures, 1)
    self.assertIn("eval_interval=-1", failures[0])

  def test_metric_compared_at_no_step_fails(self):
    reference = self._path("reference.txt", _rows([3.0, 2.9]))
    no_grad_norm = [{"step": row["step"], "learning/loss": row["learning/loss"]} for row in _rows([3.0, 2.9])]
    candidate = self._path("candidate.txt", no_grad_norm)
    reference_metrics, candidate_metrics = checks.read_metrics(reference), checks.read_metrics(candidate)
    check = checks.check_envelope("learning/grad_norm", [reference_metrics], candidate_metrics)
    self.assertEqual(check.steps_compared, 0)
    self.assertFalse(check.passed)
    failures = checks.run_checks([reference], candidate)
    self.assertIn("learning/total_weights was compared at 0 steps.", failures)
    self.assertIn("learning/grad_norm was compared at 0 steps.", failures)
    self.assertEqual(self._main(reference, candidate), 1)

  def test_non_finite_values_fail(self):
    reference_a = checks.read_metrics(self._path("a.txt", _rows([3.0, 2.9, 2.8])))
    reference_b = checks.read_metrics(self._path("b.txt", _rows([3.01, 2.92, 2.8])))
    good = checks.read_metrics(self._path("good.txt", _rows([3.0, 2.9, 2.8])))
    nan_candidate = checks.read_metrics(self._path("nan.txt", _rows([3.0, float("nan"), 2.8])))
    inf_reference = checks.read_metrics(self._path("inf.txt", _rows([3.0, float("inf"), 2.8])))
    self.assertTrue(checks.check_envelope("learning/loss", [reference_a, reference_b], good).passed)
    check = checks.check_envelope("learning/loss", [reference_a, reference_b], nan_candidate)
    self.assertFalse(check.passed)
    self.assertEqual(check.non_finite_steps, (1,))
    self.assertTrue(math.isnan(checks.noise_floor("learning/loss", [reference_a, inf_reference])))
    self.assertFalse(checks.check_envelope("learning/loss", [reference_a, inf_reference], good).passed)
    self.assertFalse(checks.check_envelope("learning/loss", [reference_a], nan_candidate).passed)
    reference = self._path("reference.txt", _rows([3.0, 2.9, 2.8]))
    failures = checks.run_checks([reference], self._path("candidate.txt", _rows([3.0, float("nan"), 2.8])))
    self.assertIn("learning/loss is not finite at steps [1] (showing up to 10).", failures)


if __name__ == "__main__":
  absltest.main()

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

"""Checks that a resized elastic run trained like the full-allocation run, from MaxText metrics files.

A resize with a constant-global-batch `elastic_global_batch_size_mode` (`constant_padded`, `constant_grad_accum`) must
keep the number of real tokens per step exactly and the optimizer mathematics unchanged; its loss and gradient norms
then differ from a full-allocation run only by floating-point ordering, i.e. by no more than two full-allocation runs
with different seeds differ from each other.
This module turns that into assertions over the per-step metrics MaxText writes with `metrics_file=`:

*   `learning/total_weights` must match the reference exactly at every common step.
*   `learning/loss` and `learning/grad_norm` must stay within an envelope: `tolerance` times the largest per-step
    difference observed between two reference runs (the seed-to-seed noise floor), or an absolute tolerance when only
    one reference run is available.

Usage from a multislice end-to-end test, after collecting the metrics files:

    python -m maxtext.tests.utils.elastic_resize_checks \
        --reference metrics_full.txt --reference metrics_full_seed2.txt --candidate metrics_resized.txt

Prints every check and exits non-zero if any of them fails.

Only training rows (those carrying `learning/*` keys) are compared. Run the reference and candidate jobs with
`eval_interval<=0` (e.g. `eval_interval=-1`): `MetricLogger` writes eval rows into the same `metrics_file`, and its
running-eval rows use steps 0, 1, ..., and the logger truncates `metrics_file` whenever it writes a row at step 0, so
every eval pass erases the training history. The checker therefore fails when it finds any `eval/*` row, when the runs
share no training steps, when a metric was compared at no step, and when a compared metric is not finite.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
from typing import TypeAlias

from absl import app
from absl import flags

_REFERENCE = flags.DEFINE_multi_string("reference", None, "Metrics file of a full-allocation run.")
_CANDIDATE = flags.DEFINE_string("candidate", None, "Metrics file of the resized run.")
_TOLERANCE = flags.DEFINE_float("tolerance", 2.0, "Multiple of the seed-to-seed noise floor allowed.")
_ABSOLUTE_TOLERANCE = flags.DEFINE_float("absolute_tolerance", 0.05, "Allowed difference with a single reference run.")

# The metrics MaxText logs for one step, e.g. `{"learning/loss": 2.9, ...}`.
StepMetrics: TypeAlias = Mapping[str, float | int]
# The metrics of one run, keyed by step.
RunMetrics: TypeAlias = dict[int, StepMetrics]

TOTAL_WEIGHTS = "learning/total_weights"
LOSS = "learning/loss"
GRAD_NORM = "learning/grad_norm"


@dataclasses.dataclass(frozen=True)
class EnvelopeCheck:
  """Result of comparing one metric of a candidate run against the reference envelope.

  Attributes:
    metric: Name of the compared metric, e.g. `learning/loss`.
    steps_compared: Number of steps at which both the candidate and the reference logged `metric`.
    max_abs_diff: Largest absolute per-step difference of `metric` between the candidate and the reference.
    allowed_abs_diff: Largest absolute per-step difference the check accepts.
    non_finite_steps: Steps at which the candidate or a reference logged a `nan` or `inf` value of `metric`.
  """

  metric: str
  steps_compared: int
  max_abs_diff: float
  allowed_abs_diff: float
  non_finite_steps: tuple[int, ...] = ()

  @property
  def passed(self) -> bool:
    """Whether `metric` was compared at some step, stayed finite and stayed within the allowed difference."""
    return self.steps_compared > 0 and not self.non_finite_steps and self.max_abs_diff <= self.allowed_abs_diff


def _read_rows(path: str) -> list[StepMetrics]:
  """Reads every row of a MaxText metrics file (one JSON object per line) that carries a `step`."""
  rows = []
  with open(path, "rt", encoding="utf8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      metrics = json.loads(line)
      if "step" in metrics:
        rows.append(metrics)
  return rows


def read_metrics(path: str) -> RunMetrics:
  """Reads the training rows of a MaxText metrics file into `{step: metrics}`.

  Only rows that carry `learning/*` keys are kept: eval and running-eval rows share the file and the `step` key and
  would otherwise replace the training row of their step. A resized run rewinds to its last snapshot after the elastic
  event and logs some steps twice; the later entry wins, which is the step the run actually continued from.
  """
  rows = _read_rows(path)
  return {int(metrics["step"]): metrics for metrics in rows if any(key.startswith("learning/") for key in metrics)}


def eval_row_steps(path: str) -> list[int]:
  """The steps of the rows of a MaxText metrics file that carry `eval/*` keys, in file order."""
  return [int(metrics["step"]) for metrics in _read_rows(path) if any(key.startswith("eval/") for key in metrics)]


def common_steps(*runs: RunMetrics) -> list[int]:
  """The steps logged by every run, in ascending order."""
  steps = set(runs[0])
  for run in runs[1:]:
    steps &= set(run)
  return sorted(steps)


def compared_steps(metric: str, reference: RunMetrics, candidate: RunMetrics) -> list[int]:
  """The common steps at which both runs logged `metric`, in ascending order."""
  steps = common_steps(reference, candidate)
  return [step for step in steps if metric in reference[step] and metric in candidate[step]]


def non_finite_steps(metric: str, runs: Sequence[RunMetrics]) -> list[int]:
  """The steps at which any of `runs` logged a `nan` or `inf` value of `metric`, in ascending order."""
  return sorted({step for run in runs for step, metrics in run.items() if not _is_finite(metrics.get(metric, 0.0))})


def _is_finite(value: float | int) -> bool:
  return math.isfinite(float(value))


def check_total_weights(reference: RunMetrics, candidate: RunMetrics) -> list[int]:
  """Returns the steps at which the candidate trained on a different number of real tokens than the reference."""
  return [
      step
      for step in compared_steps(TOTAL_WEIGHTS, reference, candidate)
      if float(reference[step][TOTAL_WEIGHTS]) != float(candidate[step][TOTAL_WEIGHTS])
  ]


def noise_floor(metric: str, references: Sequence[RunMetrics]) -> float | None:
  """Largest per-step difference of `metric` between any two reference runs, or None with fewer than two runs.

  Returns `nan` when a compared reference value is not finite: such a floor admits no candidate.
  """
  if len(references) < 2:
    return None
  floor = 0.0
  for i, ref_a in enumerate(references):
    for ref_b in references[i + 1 :]:
      for step in compared_steps(metric, ref_a, ref_b):
        value_a, value_b = float(ref_a[step][metric]), float(ref_b[step][metric])
        if not (math.isfinite(value_a) and math.isfinite(value_b)):
          return math.nan
        floor = max(floor, abs(value_a - value_b))
  return floor


def check_envelope(
    metric: str,
    references: Sequence[RunMetrics],
    candidate: RunMetrics,
    tolerance: float = 2.0,
    absolute_tolerance: float = 0.05,
) -> EnvelopeCheck:
  """Compares `metric` of the candidate against the first reference, allowing `tolerance` times the noise floor.

  With a single reference run there is no measured noise floor and `absolute_tolerance` is used instead. A `nan` or
  `inf` value of `metric` in any run fails the check, as does a metric that the runs never logged at a common step.
  """
  reference = references[0]
  non_finite = tuple(non_finite_steps(metric, [*references, candidate]))
  floor = noise_floor(metric, references)
  if floor is None:
    allowed = absolute_tolerance
  elif math.isnan(floor):
    allowed = 0.0
  else:
    allowed = max(tolerance * floor, 1e-12)
  steps = compared_steps(metric, reference, candidate)
  max_diff = 0.0
  for step in steps:
    diff = abs(float(reference[step][metric]) - float(candidate[step][metric]))
    if math.isfinite(diff):
      max_diff = max(max_diff, diff)
  return EnvelopeCheck(
      metric=metric,
      steps_compared=len(steps),
      max_abs_diff=max_diff,
      allowed_abs_diff=allowed,
      non_finite_steps=non_finite,
  )


def run_checks(
    reference_paths: Sequence[str],
    candidate_path: str,
    tolerance: float = 2.0,
    absolute_tolerance: float = 0.05,
    metrics: tuple[str, ...] = (LOSS, GRAD_NORM),
) -> list[str]:
  """Runs every check and returns the list of failure messages (empty when the candidate passes)."""
  failures = []
  for path in [*reference_paths, candidate_path]:
    eval_steps = eval_row_steps(path)
    if eval_steps:
      failures.append(
          f"{path} has eval/* rows at steps {eval_steps[:10]} (showing up to 10); eval passes can truncate the training"
          " history in the metrics file, so rerun with eval_interval=-1."
      )
  references = [read_metrics(path) for path in reference_paths]
  candidate = read_metrics(candidate_path)
  if not common_steps(references[0], candidate):
    return [*failures, "The candidate and the reference share no training steps."]
  if not compared_steps(TOTAL_WEIGHTS, references[0], candidate):
    failures.append(f"{TOTAL_WEIGHTS} was compared at 0 steps.")
  mismatched = check_total_weights(references[0], candidate)
  if mismatched:
    failures.append(f"{TOTAL_WEIGHTS} differs from the reference at steps {mismatched[:10]} (showing up to 10).")
  for metric in metrics:
    check = check_envelope(metric, references, candidate, tolerance, absolute_tolerance)
    status = "ok" if check.passed else "FAILED"
    print(
        f"{status}: {metric}: max |candidate - reference| = {check.max_abs_diff:.6g} over {check.steps_compared}"
        f" steps, allowed {check.allowed_abs_diff:.6g}, non-finite at steps {list(check.non_finite_steps[:10])}"
    )
    if check.steps_compared == 0:
      failures.append(f"{metric} was compared at 0 steps.")
    if check.non_finite_steps:
      failures.append(f"{metric} is not finite at steps {list(check.non_finite_steps[:10])} (showing up to 10).")
    if check.max_abs_diff > check.allowed_abs_diff:
      failures.append(f"{metric} exceeds the envelope: {check.max_abs_diff:.6g} > {check.allowed_abs_diff:.6g}.")
  return failures


def main(argv: Sequence[str]) -> int:
  """Command-line entry point; returns the process exit code (1 when any check fails)."""
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")
  failures = run_checks(_REFERENCE.value, _CANDIDATE.value, _TOLERANCE.value, _ABSOLUTE_TOLERANCE.value)
  for failure in failures:
    print(f"FAILED: {failure}")
  return 1 if failures else 0


if __name__ == "__main__":
  flags.mark_flags_as_required(["reference", "candidate"])
  app.run(main)

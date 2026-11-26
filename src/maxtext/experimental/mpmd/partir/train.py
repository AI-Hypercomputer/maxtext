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
r"""End-to-end training benchmark of the MPMD pipeline on Shardy MPMD and Pathways.

Lowers and compiles a microbatched training step partitioned across TPU stage
meshes by Shardy MPMD (OSS PartIR:MPMD), runs it on a Pathways backend, and
reports step time, throughput and, optionally, an XProf trace. The program
compiles only through Pathways' IFRT proxy (``JAX_PLATFORMS=proxy``), e.g. on a
TPU VM with Pathways in docker (``local_pathways.sh``)::

    JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
      python3 -m maxtext.experimental.mpmd.partir.train --pp=4 \
      --num_microbatches=4 --per_device_batch_size=4 --model_name=simple

or on a GKE cluster, as the workload ``launch.py`` prints.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags
import jax
import pathwaysutils

from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import benchmark
from maxtext.experimental.mpmd.partir import hparams as hparams_lib
from maxtext.experimental.mpmd.partir import runner as runner_lib
from maxtext.experimental.mpmd.partir import train_step as train_step_lib


def define_flags(flag_values: flags.FlagValues = flags.FLAGS) -> None:
  """Registers the trainer's flags, the hyperparameters' included, on `flag_values`."""
  hparams_lib.define_flags(flag_values)
  fv = {"flag_values": flag_values}
  flags.DEFINE_boolean(
      "enable_profile",
      False,
      "Capture an XProf / JAX profiler trace over --profile_steps extra steps run after the timed steps, so it does"
      " not affect the reported step times.",
      **fv,
  )
  flags.DEFINE_integer("profile_steps", 3, "Number of extra steps to run under the profiler once timing is done.", **fv)
  flags.DEFINE_string(
      "profile_dir",
      "",
      "Trace directory; the trace goes to <profile_dir>/<run_name>. Under Pathways it must be gs://..."
      " (pathwaysutils rejects local paths); empty skips the trace there. Elsewhere it defaults to /tmp/xprof.",
      **fv,
  )
  flags.DEFINE_float(
      "profile_delay_s",
      1.0,
      "Seconds between starting the trace and the profiled steps. Pathways workers start their TPU tracer late;"
      " without a delay the trace has no TPU device planes.",
      **fv,
  )
  flags.DEFINE_float(
      "peak_tflops_per_device",
      0.0,
      "Hardware peak TFLOP/s per JAX device (on TPU7x, half a chip's peak: a chip is two devices). When > 0 an MFU"
      " figure is reported.",
      **fv,
  )
  flags.DEFINE_string(
      "run_name", "mpmd-partir", "Name of the run: the trace directory and the label of the results.", **fv
  )
  flags.DEFINE_string("result_json", "", "If set, also write the MPMD_RESULT record (JSON) to this local path.", **fv)


def _publish_results(urls: Sequence[str]) -> None:
  """Logs where the trace went and how to open it.

  The record of a run is its ``MPMD_RESULT`` line (logged to stderr; its JSON
  also in ``--result_json``) plus the trace directory.

  Args:
    urls: Trace directories returned by :func:`benchmark.profile`.
  """
  if not urls:
    max_logging.warning("No trace was captured.")
    return
  for url in urls:
    max_logging.log(f"Trace directory     : {url}")
  if urls[0].startswith("gs://"):
    # XProf aborts on a gs:// logdir; copy the session locally first.
    max_logging.log(
        f"Open with: gcloud storage cp -r {urls[0]} /tmp/ && xprof --logdir"
        f" /tmp/{urls[0].rstrip('/').rsplit('/', 1)[-1]}"
    )


@dataclasses.dataclass(frozen=True)
class TrainingResult:
  """Everything one end-to-end MPMD training run produced.

  Attributes:
    stats: Measured step times and the final loss.
    throughput: Derived tokens/s and TFLOP/s figures.
    xprof_urls: Trace directories, empty when profiling was disabled.
    description: Human readable summary of the configuration that ran.
    num_devices: Total devices the run occupied.
    devices_per_stage: Devices backing a single pipeline stage.
    num_stages: Pipeline depth.
    num_microbatches: Microbatches per optimizer step.
    learning_rate: The AdamW learning rate the steps ran with.
  """

  stats: benchmark.StepStats
  throughput: benchmark.Throughput
  xprof_urls: tuple[str, ...]
  description: str
  num_devices: int
  devices_per_stage: int
  num_stages: int
  num_microbatches: int
  learning_rate: float = 0.0

  @property
  def analytic_bubble_fraction(self) -> float:
    """Bubble fraction a non-circular schedule cannot avoid.

    For a GPipe-style schedule the warm-up and cool-down phases leave
    ``stages - 1`` slots idle out of ``stages - 1 + microbatches``. Measured
    TensorCore idle time well above this points at transfer stalls rather than
    the schedule itself.

    Returns:
      The analytic bubble fraction, or 0.0 when it is not defined.
    """
    return analytic_bubble_fraction(self.num_stages, self.num_microbatches)

  def summary(self, label: str, schedule: str) -> str:
    """Returns the human readable results block for this run.

    The exact same text is logged and, with ``--result_json``,
    summarised in the result file.

    Args:
      label: Run label, e.g. ``mpmd-partir``.
      schedule: The pipeline schedule that ran.

    Returns:
      A multi-line summary.
    """
    stats = self.stats
    return "\n".join(
        (
            f"Results for {label} ({schedule}):",
            (f"  devices / stages / microbatches: {self.num_devices} /" f" {self.num_stages} / {self.num_microbatches}"),
            f"  steps timed             : {stats.num_steps}",
            (f"  step time mean / median : {stats.mean_ms:.3f} /" f" {stats.median_ms:.3f} ms"),
            (f"  step time min / stdev   : {stats.min_ms:.3f} /" f" {stats.stdev_ms:.3f} ms"),
            (f"  first / final loss      : {stats.first_loss:.6f} /" f" {stats.loss:.6f}"),
            *benchmark.loss_warning(stats, self.learning_rate),
            ("  analytic bubble         :" f" {self.analytic_bubble_fraction * 100:.1f}%"),
            self.throughput.report(),
        )
    )


def analytic_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
  """Returns ``(P - 1) / (P - 1 + M)`` for P stages and M microbatches.

  Args:
    num_stages: Pipeline depth.
    num_microbatches: Microbatches per optimizer step.

  Returns:
    The analytic bubble fraction, or 0.0 when it is not defined.
  """
  if num_stages <= 1 or num_microbatches <= 0:
    return 0.0
  return (num_stages - 1) / (num_stages - 1 + num_microbatches)


def run_training(
    hparams: hparams_lib.Hparams,
    *,
    enable_profile: bool = False,
    profile_dir: str = "",
    peak_tflops_per_device: float = 0.0,
    profile_tag: str = "mpmd-partir",
    profile_steps: int = 3,
    profile_delay_s: float = 1.0,
) -> TrainingResult:
  """Lowers, compiles and runs the MPMD training step, timing the steps.

  This is the whole experiment, minus flag parsing and artifact publishing, so
  that it can also be driven from a test on real accelerators.

  Args:
    hparams: The configuration to run.
    enable_profile: Whether to capture an XProf / JAX profiler trace.
    profile_dir: Trace directory (``gs://...`` under Pathways).
    peak_tflops_per_device: Hardware peak TFLOP/s per JAX device, enabling an MFU figure.
    profile_tag: Tag attached to the captured trace.
    profile_steps: Number of extra steps to run under the profiler, after the
      timed steps have finished. These are excluded from the reported timings so
      that enabling profiling cannot change the measured numbers.
    profile_delay_s: Seconds between starting the trace and the profiled steps.

  Returns:
    The :class:`TrainingResult`.
  """
  runnable = runner_lib.build(hparams, mode=train_step_lib.ExecutionMode.MPMD, check_microbatches=True)
  note = hparams_lib.ragged_buffer_note(hparams)
  if note:
    max_logging.log(f"Note: {note}")

  description = runnable.describe()
  max_logging.log("=" * 72)
  max_logging.log(description)
  max_logging.log("=" * 72)

  max_logging.log("Lowering MPMD training step...")
  lowered = runnable.lower()

  max_logging.log("Compiling MPMD training step...")
  compiled = lowered.compile()
  max_logging.log("Compilation complete.")

  # `function_named_shardings` reports where Shardy decided each input must
  # live once the program is partitioned; committing the arrays there avoids a
  # silent host round-trip on the first step.
  state, batch = runner_lib.place_on_stage_meshes(
      (runnable.state, runnable.batch),
      lowered.function_named_shardings.input_specs,
  )

  # The profiler is deliberately NOT wrapped around the timed steps. XProf adds
  # per-executable tracing overhead, so profiling the measured window would
  # inflate every number this run reports -- and silently, because the result
  # still looks plausible. Instead the timed steps run clean, and the trace is
  # captured afterwards from a short extra window that nothing is derived from.
  # MaxText does the equivalent with `skip_first_n_steps_for_profiler`.
  state, stats = benchmark.run_steps(
      compiled,
      state,
      batch,
      num_steps=hparams.steps,
      warmup_steps=hparams.warmup_steps,
      abort_on_nonfinite=True,
  )

  urls: list[str] = []
  if enable_profile:
    max_logging.log(
        f"Capturing XProf over {profile_steps} extra step(s); these are not" " included in the reported timings."
    )
    with benchmark.profile(
        True,
        tag=profile_tag,
        profile_dir=profile_dir,
        trace_delay_s=profile_delay_s,
    ) as captured:
      benchmark.run_steps(
          compiled,
          state,
          batch,
          num_steps=profile_steps,
          warmup_steps=0,
          first_step=hparams.warmup_steps + hparams.steps,
          log_prefix="[profiled] ",
          abort_on_nonfinite=True,
      )
    urls = list(captured)

  throughput = benchmark.compute_throughput(
      stats=stats,
      maxtext_config=runnable.maxtext_config,
      devices_per_stage=runnable.plan.topology.devices_per_stage,
      num_devices=runnable.plan.num_devices,
      peak_tflops_per_device=peak_tflops_per_device,
  )

  return TrainingResult(
      stats=stats,
      throughput=throughput,
      xprof_urls=tuple(urls),
      description=description,
      num_devices=runnable.plan.num_devices,
      devices_per_stage=runnable.plan.topology.devices_per_stage,
      num_stages=hparams.num_stages,
      num_microbatches=hparams.num_microbatches,
      learning_rate=hparams.learning_rate,
  )


def result_line(hparams: hparams_lib.Hparams, result: TrainingResult, run_name: str = "") -> str:
  """Returns the one-line ``MPMD_RESULT {json}`` record of a finished run.

  One JSON object per run, so results can be collected from logs without
  scraping the human-readable summary.

  Args:
    hparams: The configuration that ran.
    result: What the run measured.
    run_name: The run's name (``--run_name``), or "" for none.

  Returns:
    ``"MPMD_RESULT "`` followed by a single-line JSON object.
  """
  throughput = result.throughput
  record = {
      "backend": "partir",
      "model": hparams.model_name,
      "layers": hparams.num_layers,
      "stages": result.num_stages,
      "microbatches": result.num_microbatches,
      "schedule": hparams.pipeline_schedule,
      "fsdp": hparams.fsdp,
      "expert": hparams.expert,
      "per_device_batch_size": hparams.per_device_batch_size,
      "seq": hparams.max_target_length,
      "remat": hparams.remat,
      "devices": result.num_devices,
      "devices_per_stage": result.devices_per_stage,
      "step_ms": throughput.step_time_s * 1e3,
      "step_ms_mean": result.stats.mean_ms,
      "tokens_per_step": throughput.tokens_per_step,
      "tokens_per_s_per_device": throughput.tokens_per_second_per_device,
      "tflops_per_device": throughput.tflops_per_second_per_device,
      "mfu": throughput.mfu,
      "loss": result.stats.loss,
      "loss_first": result.stats.first_loss,
      "learning_rate": hparams.learning_rate,
  }
  if hparams.maxtext_overrides:
    record["maxtext_overrides"] = list(hparams.maxtext_overrides)
  if run_name:
    record["run"] = run_name
  return "MPMD_RESULT " + json.dumps(record, sort_keys=True, default=str)


def _write_result(line: str, path: str) -> None:
  """Logs the ``MPMD_RESULT`` line, and writes its JSON to `path` if set."""
  max_logging.log(line)
  if path:
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(line.removeprefix("MPMD_RESULT ") + "\n")


def main(argv: Sequence[str], flag_values: flags.FlagValues = flags.FLAGS) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected command line arguments: {argv[1:]}")
  # Registers the Pathways proxy backend when JAX_PLATFORMS contains "proxy";
  # it must run before the backend is first used.
  pathwaysutils.initialize()

  fv = flag_values
  hparams = hparams_lib.from_flags(fv)
  try:
    result = run_training(
        hparams,
        enable_profile=fv.enable_profile,
        profile_dir=fv.profile_dir,
        peak_tflops_per_device=fv.peak_tflops_per_device,
        profile_tag=fv.run_name,
        profile_steps=fv.profile_steps,
        profile_delay_s=fv.profile_delay_s,
    )
  except benchmark.NonFiniteLossError as error:
    # Without this a NaN loss ended in "Execution finished successfully" with
    # exit 0, after steps whose times described degenerate routing.
    record = {"backend": "partir", "model": hparams.model_name, "error": f"non-finite loss: {error}"}
    if fv.run_name:
      record["run"] = fv.run_name
    _write_result("MPMD_RESULT " + json.dumps(record, sort_keys=True), fv.result_json)
    raise SystemExit(f"MPMD training failed: {error}") from error
  summary = result.summary(fv.run_name, hparams.pipeline_schedule)

  max_logging.log("=" * 72)
  max_logging.log(summary)
  max_logging.log("=" * 72)
  _write_result(result_line(hparams, result, fv.run_name), fv.result_json)
  _publish_results(result.xprof_urls)
  max_logging.log("Execution finished successfully.")


if __name__ == "__main__":
  define_flags()
  jax.config.config_with_absl()
  app.run(main)

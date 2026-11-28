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
"""Step timing, throughput accounting and XProf capture.

Kept separate from the trainer so the measurement logic can be unit tested and
so the SPMD comparison reports exactly the same quantities as the MPMD run.

The throughput numbers are deliberately reported in the same units MaxText
itself logs (``TFLOP/s/device`` and ``Tokens/s/device``, see
``maxtext/src/maxtext/common/metric_logger.py``) so that an MPMD run can be
compared against a MaxText SPMD pipeline run without any unit conversion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import math
import statistics
import time
from typing import Any

import jax

from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import hparams as hparams_lib


class NonFiniteLossError(RuntimeError):
  """A step returned a NaN or infinite loss (see :func:`run_steps`)."""

  def __init__(self, step: int, loss: float, losses: tuple[float, ...]):
    # The losses end at `step` and may start after step 1 (first_step).
    super().__init__(
        f"step {step} returned a loss of {loss} (losses from step"
        f" {step - len(losses) + 1}:"
        f" {', '.join(f'{value:.6g}' for value in losses)})"
    )
    self.step = step
    self.loss = loss
    self.losses = losses


@dataclasses.dataclass(frozen=True)
class StepStats:
  """Timing statistics for a sequence of training steps.

  Attributes:
    step_times_ms: Wall-clock duration of each timed step.
    loss: Loss reported by the final step.
    first_loss: Loss reported by the first step run (warm-up included).
  """

  step_times_ms: tuple[float, ...]
  loss: float
  first_loss: float = float("nan")

  @property
  def num_steps(self) -> int:
    """Returns the number of timed steps."""
    return len(self.step_times_ms)

  @property
  def mean_ms(self) -> float:
    """Returns the arithmetic mean step time in milliseconds."""
    return statistics.fmean(self.step_times_ms) if self.step_times_ms else 0.0

  @property
  def median_ms(self) -> float:
    """Returns the median step time in milliseconds."""
    return statistics.median(self.step_times_ms) if self.step_times_ms else 0.0

  @property
  def min_ms(self) -> float:
    """Returns the minimum step time in milliseconds."""
    return min(self.step_times_ms) if self.step_times_ms else 0.0

  @property
  def stdev_ms(self) -> float:
    """Returns the sample standard deviation of step times in milliseconds."""
    if len(self.step_times_ms) < 2:
      return 0.0
    return statistics.stdev(self.step_times_ms)


def loss_warning(stats: StepStats, learning_rate: float) -> list[str]:
  """Returns a warning when the loss rose over the run or is not finite.

  Every step trains on the same synthetic batch, so a healthy run's loss falls;
  a rise from the first step to the last means the optimizer diverged.
  llama3.1-8b at the default learning rate of 1e-3 went from 24.8 to 675 in 7
  steps (and trained with 1e-5). The step times stay valid, but the run is not
  training.

  Args:
    stats: The timed steps' statistics.
    learning_rate: The learning rate the steps ran with.
  """
  if stats.num_steps and not math.isfinite(stats.loss):
    return [
        f"  ERROR                   : the loss is {stats.loss} after"
        f" {stats.num_steps} timed steps (first {stats.first_loss:.4g}); the"
        " run did not train, and its step times are not valid"
    ]
  if not stats.loss > stats.first_loss:  # also False when either is NaN
    return []
  return [
      f"  WARNING                 : the loss rose from {stats.first_loss:.4g}"
      f" to {stats.loss:.4g}: --learning_rate {learning_rate:g} is too high"
      " for this model (llama3.1-8b diverges at 1e-3 and trains at 1e-5)"
  ]


@dataclasses.dataclass(frozen=True)
class Throughput:
  """Derived performance figures for one training step.

  Attributes:
    step_time_s: Median step duration, in seconds.
    tokens_per_step: Number of training tokens consumed per step.
    num_devices: Total devices participating in the run.
    tflops_per_step_per_device: Analytic model FLOPs per device per step.
    peak_tflops_per_device: Hardware peak, or 0 when unknown.
  """

  step_time_s: float
  tokens_per_step: int
  num_devices: int
  tflops_per_step_per_device: float
  peak_tflops_per_device: float = 0.0

  @property
  def tokens_per_second(self) -> float:
    """Returns global training throughput in tokens per second."""
    return self.tokens_per_step / self.step_time_s if self.step_time_s else 0.0

  @property
  def tokens_per_second_per_device(self) -> float:
    """Returns per-device training throughput in tokens per second."""
    if not self.num_devices:
      return 0.0
    return self.tokens_per_second / self.num_devices

  @property
  def tflops_per_second_per_device(self) -> float:
    """Returns achieved TFLOP/s per device."""
    if not self.step_time_s:
      return 0.0
    return self.tflops_per_step_per_device / self.step_time_s

  @property
  def mfu(self) -> float | None:
    """Model FLOPs utilization, or None if the hardware peak is unknown."""
    if self.peak_tflops_per_device <= 0:
      return None
    return self.tflops_per_second_per_device / self.peak_tflops_per_device

  def report(self) -> str:
    """Returns the figures as aligned ``key: value`` lines."""
    lines = [
        f"step_time_ms            : {self.step_time_s * 1e3:.3f}",
        f"tokens_per_step         : {self.tokens_per_step}",
        f"num_devices             : {self.num_devices}",
        f"Tokens/s                : {self.tokens_per_second:,.1f}",
        f"Tokens/s/device         : {self.tokens_per_second_per_device:,.1f}",
        f"TFLOP/s/device          : {self.tflops_per_second_per_device:.3f}",
    ]
    mfu = self.mfu
    if mfu is not None:
      lines.append(f"MFU                     : {mfu * 100:.2f}%")
    return "\n".join(f"  {line}" for line in lines)


def compute_throughput(
    *,
    stats: StepStats,
    maxtext_config: Any,
    devices_per_stage: int,
    num_devices: int,
    peak_tflops_per_device: float = 0.0,
) -> Throughput:
  """Derives throughput figures from `stats`.

  MaxText's analytic FLOP counter reports FLOPs for ``per_device_batch_size``
  tokens across every layer. In this pipeline the global batch is
  ``per_device_batch_size * devices_per_stage`` (the batch is data-parallel
  over one stage, not over the whole slice), so the whole-step model FLOPs are
  the MaxText figure scaled by `devices_per_stage`. Dividing by the *total*
  device count then charges the pipeline for the devices it is occupying,
  including any that are idle in a bubble -- which is the honest way to
  compare a pipeline against a non-pipelined baseline.

  The FLOPs are those of the model the MPMD step runs: MaxText also counts a
  multi-token-prediction block's, which the MPMD model does not build
  (``config.py`` refuses ``mtp_num_layers > 0``), so they are never charged.

  Args:
    stats: Measured step times.
    maxtext_config: The MaxText config the run used.
    devices_per_stage: Number of devices backing one pipeline stage.
    num_devices: Total number of devices in the run.
    peak_tflops_per_device: Hardware peak TFLOP/s, 0 if unknown.

  Returns:
    The derived :class:`Throughput`.
  """
  # Imported here so that modules which only need the dataclasses above do not
  # pay for MaxText's import graph.
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

  if maxtext_config.mtp_num_layers > 0:
    maxtext_config = maxtext_config.replace(mtp_num_layers=0)
  maxtext_tflops_per_device, _, _ = maxtext_utils.calculate_tflops_training_per_device(maxtext_config, log=False)
  total_tflops_per_step = maxtext_tflops_per_device * devices_per_stage
  tokens_per_step = int(maxtext_config.per_device_batch_size * devices_per_stage * maxtext_config.max_target_length)
  return Throughput(
      step_time_s=stats.median_ms / 1e3,
      tokens_per_step=tokens_per_step,
      num_devices=num_devices,
      tflops_per_step_per_device=total_tflops_per_step / max(num_devices, 1),
      peak_tflops_per_device=peak_tflops_per_device,
  )


def run_steps(
    step_fn: Callable[..., Any],
    state: Any,
    batch: Any,
    *,
    num_steps: int,
    warmup_steps: int = 0,
    first_step: int = 0,
    log_prefix: str = "",
    abort_on_nonfinite: bool = False,
) -> tuple[Any, StepStats]:
  """Runs `step_fn` repeatedly, timing the steps after the warm-up.

  Args:
    step_fn: A compiled ``(state, batch) -> (state, loss)``.
    state: Initial state, threaded through the steps.
    batch: The (fixed) input batch.
    num_steps: Number of timed steps.
    warmup_steps: Number of untimed steps run first. The first execution pays
      for dispatch-path warm-up and, on Pathways, for program distribution;
      including it would dominate the mean.
    first_step: Steps the state has been through in earlier calls, so that
      :class:`NonFiniteLossError` numbers the step among all of them (it lists
      this call's losses only): a NaN in train.py's second profiled step after
      one warm-up and three timed steps is step 6, not "step 2".
    log_prefix: Prefix for the per-step log lines.
    abort_on_nonfinite: Raise :class:`NonFiniteLossError` at the first NaN or
      infinite loss (warm-up included) instead of running on: every later step
      is wasted, and its timing describes a degenerate run.

  Returns:
    A tuple of the final state and the collected :class:`StepStats`.

  Raises:
    NonFiniteLossError: See ``abort_on_nonfinite``.
  """
  loss_value = first_loss = float("nan")
  losses: list[float] = []

  def check(step: int) -> None:
    losses.append(loss_value)
    if abort_on_nonfinite and not math.isfinite(loss_value):
      raise NonFiniteLossError(first_step + step, loss_value, tuple(losses))

  # Every step is wrapped in a `StepTraceAnnotation`. XProf uses these
  # annotations -- and nothing else -- to segment a trace into training steps.
  # Without them an XProf session reports `is_training: false` and a
  # `steptime_ms_average` derived from the whole (mostly idle) profile window,
  # which is not the step time of this trainer. MaxText emits the same
  # annotation with the same "train" name, so keeping it identical here is what
  # makes an MPMD XProf directly comparable to a MaxText SPMD one.
  # `block_until_ready` must cover the updated state, not just the loss. The
  # loss is produced part-way through the step, while the optimizer update
  # finishes later; blocking on the loss alone would let the tail of step N
  # overlap step N+1 and be attributed to it.
  for index in range(warmup_steps):
    with jax.profiler.StepTraceAnnotation("train", step_num=index):
      state, loss = step_fn(state, batch)
      jax.block_until_ready((state, loss))
    loss_value = float(loss)
    if index == 0:
      first_loss = loss_value
    max_logging.log(f"{log_prefix}warmup step {index + 1}/{warmup_steps}" f" loss={loss_value:.6f}")
    check(index + 1)

  step_times_ms = []
  for index in range(num_steps):
    started = time.perf_counter()
    with jax.profiler.StepTraceAnnotation("train", step_num=warmup_steps + index):
      state, loss = step_fn(state, batch)
      jax.block_until_ready((state, loss))
    elapsed_ms = (time.perf_counter() - started) * 1e3
    step_times_ms.append(elapsed_ms)
    loss_value = float(loss)
    if index == 0 and not warmup_steps:
      first_loss = loss_value
    max_logging.log(
        f"{log_prefix}completed step: {index + 1}/{num_steps},"
        f" seconds: {elapsed_ms / 1e3:.4f}, loss: {loss_value:.6f}"
    )
    check(warmup_steps + index + 1)

  return state, StepStats(
      step_times_ms=tuple(step_times_ms),
      loss=loss_value,
      first_loss=first_loss,
  )


@contextlib.contextmanager
def profile(
    enabled: bool,
    *,
    tag: str,
    profile_dir: str = "",
    trace_delay_s: float = 1.0,
) -> Iterator[list[str]]:
  """Captures an XProf trace around the enclosed block.

  On a Pathways backend ``pathwaysutils`` turns ``jax.profiler.start_trace``
  into a Pathways profile request: the workers write the trace themselves, so
  `profile_dir` must be a ``gs://`` directory (a local one is rejected). On any
  other backend the trace goes to `profile_dir` (default ``/tmp/xprof``).
  Either way it lands in ``<profile_dir>/<tag>``.

  Pathways workers start their TPU tracer late: device work issued right after
  ``start_trace`` returns is missing from the trace (no ``/device:TPU`` planes).
  The block therefore starts `trace_delay_s` after the trace does.

  The trace covers one TPU, one SparseCore and one SparseCore tile a traced
  host (`profile_options`); a backend that refuses those options gets an
  untrimmed trace instead of none.

  Args:
    enabled: When False this is a no-op, and the yielded list stays empty.
    tag: Name of the trace directory under `profile_dir`.
    profile_dir: Output directory; must be ``gs://...`` under Pathways.
    trace_delay_s: Seconds to wait after ``start_trace`` on Pathways.

  Yields:
    A list that will contain the trace directory once the block has exited.
  """
  urls: list[str] = []
  if not enabled:
    yield urls
    return

  # Imported lazily so CPU unit tests do not need pathwaysutils.
  import pathwaysutils  # pylint: disable=import-outside-toplevel

  on_pathways = bool(pathwaysutils.is_pathways_backend_used())
  root = profile_dir or ("" if on_pathways else "/tmp/xprof")
  if on_pathways and not root.startswith("gs://"):
    max_logging.warning(
        "Skipping the XProf trace: under Pathways jax.profiler.start_trace"
        f" accepts only gs:// directories, got --profile_dir={profile_dir!r}."
    )
    yield urls
    return
  destination = f"{root.rstrip('/')}/{tag}"
  max_logging.log(f"Starting JAX profiler trace to {destination}...")
  try:
    jax.profiler.start_trace(destination, profiler_options=profile_options())
  except Exception as error:  # pylint: disable=broad-exception-caught
    max_logging.warning(f"The trimmed trace was refused ({error}); tracing untrimmed.")
    jax.profiler.start_trace(destination)
  try:
    if on_pathways and trace_delay_s > 0:
      time.sleep(trace_delay_s)
    yield urls
  except BaseException:
    with contextlib.suppress(Exception):
      jax.profiler.stop_trace()
    raise
  # The measured steps have already run when the trace is written, so a trace
  # that fails to write is reported, not raised: discarding a completed run
  # because its profile could not be saved would be backwards.
  try:
    jax.profiler.stop_trace()
  except Exception as error:  # pylint: disable=broad-exception-caught
    max_logging.warning(f"XProf trace could not be written: {error}")
    return
  urls.append(destination)
  max_logging.log(f"JAX profiler trace written to {destination}.")


def profile_options() -> jax.profiler.ProfileOptions:
  """`jax.profiler.ProfileOptions` for `hparams.TRACE_ADVANCED_CONFIGURATION`."""
  options = jax.profiler.ProfileOptions()
  options.advanced_configuration = dict(hparams_lib.TRACE_ADVANCED_CONFIGURATION)
  return options

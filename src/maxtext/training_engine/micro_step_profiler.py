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

"""xprof profiling for the training engine, windowed in micro-steps.

The pre-train profiler in `maxtext.common.profiler` counts optimizer steps. That unit does
not survive the move to RL: the engine does not own the training loop, the orchestrator
decides where the step boundaries fall, and with sequence packing a single step can hold
hundreds of micro-batches -- a five-step window would produce a trace too large to open.

So the profiler config keys keep their names and change their unit. Here
`skip_first_n_steps_for_profiler`, `profiler_steps` and `profile_periodically_period` all
count *micro-steps*, i.e. `fwd_bwd` calls, which makes `profiler_steps` a direct control on
trace size. Windows are placed against the engine's cumulative micro-step counter, which is
checkpointed, so a window fires once over the lifetime of a run rather than once per restart.

A window will usually not contain an optimizer update: 5 micro-steps out of the few hundred
in an RL step rarely spans one. Sizing `profiler_steps` past the micro-steps-per-step count
is the way to capture one, at the cost of a correspondingly large trace. The summary logged
when a window closes reports what it actually covered, so this is diagnosable rather than
mysterious.

This is a separate class from `maxtext.common.profiler.Profiler` rather than a mode of it:
the two share only their config keys and their `ProfileOptions` builder (imported from there),
and differ on unit, on failure policy -- a misconfigured window here warns and disables rather
than raising, because an RL job must not die over a diagnostic -- on the Pathways `max_num_hosts`
handling, and on the device drains an asynchronously dispatched window needs. Merging them
would be a class with two disjoint halves behind a mode flag.

Two pre-train profiling backends are deliberately absent rather than unfinished. nsys is a
CUDA path and the engine is Pathways-only, which is TPU. Managed mldiagnostics is a pre-train
integration nobody has asked the engine for; setting it warns rather than silently writing
traces somewhere the operator is not looking for them. Both are additions to make when
something needs them, not gaps to fill in.
"""

from typing import Any, Callable

from absl import logging
import jax

from maxtext.common.profiler import build_profile_options
from maxtext.configs import pyconfig


class MicroStepProfiler:
  """Activates and deactivates xprof over a window measured in micro-steps.

  The caller drives this from `fwd_bwd`, passing the cumulative micro-step index that is
  about to run and, once it has, the same index again. Everything else -- whether profiling
  is on at all, where the window sits, whether to drain the device at its edges -- is read
  off the config.
  """

  def __init__(self, config: pyconfig.HyperParameters) -> None:
    """Builds the profiler and resolves the window from the config.

    Never raises on a bad window: profiling is a diagnostic, and an RL job that would
    otherwise train fine must not die because the window was misconfigured. Anything wrong
    disables profiling with a warning.

    Args:
      config: MaxText configuration. Read for `profiler`, `tensorboard_dir`, and the
        `*_profiler*` keys, whose step units are reinterpreted as micro-steps.
    """
    self._enabled = False
    self._active = False
    # First micro-step of the currently open window, for the output subdirectory name and
    # for the summary logged when it closes.
    self._window_start: int | None = None
    self._compiled_in_window = False

    # `config.profiler` is a `ProfilerType`, whose `str()` is "ProfilerType.XPLANE" rather
    # than the value; take `.value` when there is one so both forms compare as the string.
    mode = getattr(config.profiler, "value", config.profiler)
    if mode == "":
      return
    if mode != "xplane":
      logging.warning(
          "profiler=%r is not supported by the training engine (Pathways/TPU runs xplane only); "
          "profiling is disabled.",
          mode,
      )
      return

    if getattr(config, "managed_mldiagnostics", False):
      # Pre-train routes xplane through the mldiagnostics collector when this is set. The
      # engine does not, so say so: otherwise the traces land in tensorboard_dir and the
      # operator is left waiting for them to show up in the managed UI.
      logging.warning(
          "managed_mldiagnostics is set, but the training engine does not route profiles through "
          "it. Traces are written directly to tensorboard_dir=%r instead.",
          config.tensorboard_dir,
      )

    self._output_dir = config.tensorboard_dir
    self._profile_cleanly = config.profile_cleanly
    self._upload_all = config.upload_all_profiler_results
    self._first = config.skip_first_n_steps_for_profiler
    self._length = config.profiler_steps
    self._period = config.profile_periodically_period

    if self._first < 0 or self._length <= 0:
      logging.warning(
          "Profiling disabled: skip_first_n_steps_for_profiler=%d and profiler_steps=%d do not "
          "describe a non-empty window. In the training engine both count micro-steps (fwd_bwd "
          "calls), not optimizer steps.",
          self._first,
          self._length,
      )
      return
    self._last = self._first + self._length - 1

    if 0 < self._period <= self._length:
      logging.warning(
          "profile_periodically_period=%d is not greater than profiler_steps=%d, so periodic "
          "windows would overlap. Falling back to a single window.",
          self._period,
          self._length,
      )
      self._period = -1

    if not self._output_dir:
      # Defensive: `pyconfig.initialize` backfills `base_output_directory` and `run_name`, so a
      # config that came through it always has this set. A `MaxTextConfig` built directly does
      # not, and the trace path would then be built out of the unset value -- `None/micro_step_0`,
      # or `/micro_step_0` at the filesystem root for an empty string.
      logging.warning(
          "Profiling disabled: tensorboard_dir=%r, so there is nowhere to write the trace. Set "
          "base_output_directory and run_name, or set tensorboard_dir directly.",
          self._output_dir,
      )
      return

    if not str(self._output_dir).startswith("gs://"):
      # Not fatal: only the Pathways backend requires GCS, and local runs (tests, McJAX
      # debugging) profile to a local directory fine. Under Pathways `start_trace` raises,
      # and `_start` catches it.
      logging.warning(
          "tensorboard_dir=%r is not a gs:// path. Profiling under Pathways requires a GCS "
          "destination and will fail to start.",
          self._output_dir,
      )

    # Managed mldiagnostics is a pre-train path the engine does not support, so the power
    # events are always ours to collect.
    self._profiling_options = build_profile_options(config)
    self._enabled = True
    logging.info(
        "Profiler enabled: capturing micro-steps %d-%d%s. Note that in the training engine "
        "skip_first_n_steps_for_profiler, profiler_steps and profile_periodically_period count "
        "micro-batches (fwd_bwd calls), not optimizer steps.",
        self._first,
        self._last,
        f", repeating every {self._period} micro-steps" if self._period > 0 else "",
    )

  @property
  def is_active(self) -> bool:
    """Returns True while a trace is open."""
    return self._active

  def note_compilation(self) -> None:
    """Records that a compilation happened, so a window containing one says so on close.

    With sequence packing the batch signature changes across micro-batches, and each change
    recompiles. A recompile landing inside a window dominates the trace, and a reader who
    does not know it happened will draw the wrong conclusion from it.
    """
    if self._active:
      self._compiled_in_window = True

  def maybe_activate(self, micro_step: int, drain: Callable[[], None] | None = None) -> None:
    """Starts a trace if `micro_step` opens a window.

    Args:
      micro_step: Cumulative index of the micro-step about to run.
      drain: Called before the trace starts when `profile_cleanly` is set, to wait out the
        computations already in flight. Without it the trace opens over work belonging to
        earlier micro-steps, since `fwd_bwd` only dispatches and returns.
    """
    if not self._enabled or self._active or not self._opens_window(micro_step):
      return
    if self._profile_cleanly and drain is not None:
      drain()
    self._start(micro_step)

  def maybe_deactivate(self, micro_step: int, block_on: Any = None, train_step: int | None = None) -> None:
    """Stops the trace if `micro_step` closes the window.

    Args:
      micro_step: Cumulative index of the micro-step that just ran.
      block_on: Awaited before the trace stops when `profile_cleanly` is set. Must be
        something the last captured micro-step produced -- the gradient accumulator, say.
        `fwd_bwd` returns once the work is enqueued, so stopping without this cuts the trace
        before the device has run what the window was supposed to capture.
      train_step: Current optimizer step, reported in the closing summary.
    """
    if not self._enabled or not self._active or not self._closes_window(micro_step):
      return
    if self._profile_cleanly and block_on is not None:
      jax.block_until_ready(block_on)
    self._stop(micro_step, train_step)

  def close(self) -> None:
    """Stops an open trace, if any.

    A run that ends or fails mid-window leaves the trace open; without this the profile is
    never written out.
    """
    if self._active:
      logging.info("Stopping an open profiler trace at shutdown.")
      self._stop(None, None)

  def _opens_window(self, micro_step: int) -> bool:
    if micro_step == self._first:
      return True
    # The `>= self._first` guard matters: Python's modulo is non-negative, so without it a
    # micro-step before the first one can satisfy the congruence and open a window early.
    return self._period > 0 and micro_step >= self._first and (micro_step - self._first) % self._period == 0

  def _closes_window(self, micro_step: int) -> bool:
    if micro_step == self._last:
      return True
    return self._period > 0 and micro_step >= self._last and (micro_step - self._last) % self._period == 0

  def _max_num_hosts(self) -> int:
    """Returns how many Pathways hosts to trace.

    `upload_all_profiler_results` is the pre-train key for this, phrased for McJAX where
    every process writes its own result. Under a single controller the equivalent knob is
    Pathways' `max_num_hosts`, which defaults to 1.
    """
    if not self._upload_all:
      return 1
    try:
      hosts = len({d.process_index for d in jax.devices()})
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Could not determine the host count (%s); profiling one host.", e)
      return 1
    return max(hosts, 1)

  def _start(self, micro_step: int) -> None:
    """Starts the trace, tolerating a backend that already has one open."""
    output_path = f"{str(self._output_dir).rstrip('/')}/micro_step_{micro_step}"
    kwargs: dict[str, Any] = {"profiler_options": self._profiling_options}
    max_num_hosts = self._max_num_hosts()
    if max_num_hosts != 1:
      # Only Pathways' patched `start_trace` accepts this; upstream JAX would reject it.
      kwargs["max_num_hosts"] = max_num_hosts
    try:
      jax.profiler.start_trace(output_path, **kwargs)
    except TypeError:
      # A non-Pathways backend, where `max_num_hosts` is not a parameter. One host it is.
      kwargs.pop("max_num_hosts", None)
      try:
        jax.profiler.start_trace(output_path, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Could not start the profiler at micro-step %d: %s", micro_step, e)
        return
    except Exception as e:  # pylint: disable=broad-except
      # Pathways allows one trace per backend, so a concurrent trace elsewhere in the RL
      # cluster lands here. Never take the training run down over it.
      logging.warning("Could not start the profiler at micro-step %d: %s", micro_step, e)
      return
    self._active = True
    self._window_start = micro_step
    self._compiled_in_window = False
    logging.info("Profiling micro-steps %d-%d to %s.", micro_step, micro_step + self._length - 1, output_path)

  def _stop(self, micro_step: int | None, train_step: int | None) -> None:
    """Stops the trace and reports what the window turned out to contain."""
    try:
      jax.profiler.stop_trace()
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Could not stop the profiler: %s", e)
    finally:
      self._active = False

    if micro_step is None:
      # `close()` stops a window the run never reached the end of, so there is no closing
      # micro-step to name -- only the one the window opened on.
      summary = f"Profile written for micro-steps {self._window_start} onwards; the window was cut short at shutdown."
    else:
      summary = f"Profile written for micro-steps {self._window_start}-{micro_step}."
    if train_step is not None:
      summary += f" Optimizer step at the close of the window: {train_step}."
    if self._profile_cleanly:
      summary += (
          " The first and last captured micro-steps are bracketed by device drains, so their"
          " timings are not representative of steady state."
      )
    if self._compiled_in_window:
      summary += " A compilation ran inside this window and will dominate the trace."
    logging.info(summary)
    self._window_start = None
    self._compiled_in_window = False

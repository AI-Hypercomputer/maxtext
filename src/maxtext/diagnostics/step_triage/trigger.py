# Copyright 2023–2026 Google LLC
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

"""Pure-stdlib step-time regression trigger with workload-event exclusions."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
import dataclasses
import statistics
from typing import Any


VALID_MODES: frozenset[str] = frozenset({"target", "baseline", "both"})

CAUSE_NONE = "none"
CAUSE_DEVICE_SLOW = "DEVICE_SLOW"
CAUSE_INPUT_BOUND = "INPUT_BOUND"

STAGE_NONE = "HEALTHY_NO_PROBE"
STAGE_A_HOST_TIMING = "STAGE_A_HOST_TIMING"
STAGE_B_METRICS = "STAGE_B_METRICS"
STAGE_C_ICI_PROBE = "STAGE_C_ICI_PROBE"
STAGE_D_DIAG_STEPS = "STAGE_D_DIAG_STEPS"

ALL_PROBE_STAGES: tuple[str, ...] = (
    STAGE_A_HOST_TIMING,
    STAGE_B_METRICS,
    STAGE_C_ICI_PROBE,
    STAGE_D_DIAG_STEPS,
)

DEFAULT_EXCLUDE_REASONS: tuple[str, ...] = (
    "profiler",
    "checkpoint",
    "eval",
    "data_wait",
    "dropless_replay",
    "recompile",
)

_CONFIGURABLE_EXCLUDE_REASONS: frozenset[str] = frozenset(DEFAULT_EXCLUDE_REASONS)


def _lookup_first(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
  for key in keys:
    if key in mapping and mapping[key] is not None:
      return mapping[key]
  return default


def _normalize_exclude_tuple(raw: Any) -> tuple[str, ...]:
  if raw is None:
    return DEFAULT_EXCLUDE_REASONS
  if isinstance(raw, str):
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(items)
  if isinstance(raw, Iterable):
    return tuple(str(item).strip() for item in raw if str(item).strip())
  return DEFAULT_EXCLUDE_REASONS


@dataclasses.dataclass
class TriggerConfig:
  """Configuration for the step-time straggler trigger.

  Attributes:
    mode: Trigger mode (`"target"`, `"baseline"`, or `"both"`).
    target_step_ms: Expected step time in ms for `"target"` or `"both"` mode.
    tolerance: Fractional threshold above `target_step_ms` (default `0.05` = +5%).
    slowdown_ratio_tolerance: Fractional threshold above the rolling baseline
      median in `"baseline"` or `"both"` mode. Defaults to `tolerance` when `None`.
    consecutive_steps: Number of consecutive clean slow steps `N` required to fire.
    warmup_steps: Number of initial steps (`0 .. warmup_steps - 1`) excluded as warmup.
    window: Rolling window size of clean steps used to compute the baseline median.
    exclude: Tuple of standard event tags that exclude a step from streak/baseline.
    exclude_after_checkpoint_steps: Number of subsequent steps (`k+1 .. k+M`)
      excluded after a checkpoint at step `k` (default `2` for async checkpointing).
    cooldown_steps: Number of steps after firing during which re-firing is suppressed.
  """

  mode: str = "target"
  target_step_ms: float | None = None
  tolerance: float = 0.05
  slowdown_ratio_tolerance: float | None = None
  consecutive_steps: int = 5
  warmup_steps: int = 10
  window: int = 20
  exclude: tuple[str, ...] = DEFAULT_EXCLUDE_REASONS
  exclude_after_checkpoint_steps: int = 2
  cooldown_steps: int = 50

  def __post_init__(self) -> None:
    self.mode = str(self.mode).strip().lower()
    if self.mode not in VALID_MODES:
      raise ValueError(f"Unsupported trigger mode {self.mode!r}; expected one of {sorted(VALID_MODES)}")
    if self.target_step_ms is not None:
      self.target_step_ms = float(self.target_step_ms)
      if self.target_step_ms <= 0:
        raise ValueError("target_step_ms must be positive when provided")
    self.tolerance = float(self.tolerance)
    if self.slowdown_ratio_tolerance is not None:
      self.slowdown_ratio_tolerance = float(self.slowdown_ratio_tolerance)
    self.consecutive_steps = max(1, int(self.consecutive_steps))
    self.warmup_steps = max(0, int(self.warmup_steps))
    self.window = max(1, int(self.window))
    self.exclude = _normalize_exclude_tuple(self.exclude)
    self.exclude_after_checkpoint_steps = max(0, int(self.exclude_after_checkpoint_steps))
    self.cooldown_steps = max(0, int(self.cooldown_steps))

  @property
  def effective_slowdown_tolerance(self) -> float:
    if self.slowdown_ratio_tolerance is not None:
      return self.slowdown_ratio_tolerance
    return self.tolerance

  @classmethod
  def from_dict(cls, cfg: Mapping[str, Any]) -> TriggerConfig:
    """Constructs a `TriggerConfig` from unprefixed or `straggler_probe_*` MaxText keys."""
    mode = _lookup_first(
        cfg,
        ("straggler_probe_trigger_mode", "straggler_probe_mode", "trigger_mode", "mode"),
        "target",
    )
    target_step_ms = _lookup_first(
        cfg,
        ("straggler_probe_target_step_ms", "straggler_probe_target_ms", "target_step_ms", "target_ms"),
        None,
    )
    tolerance = _lookup_first(
        cfg,
        ("straggler_probe_tolerance", "straggler_probe_target_tolerance", "tolerance"),
        0.05,
    )
    slowdown_tol = _lookup_first(
        cfg,
        ("straggler_probe_slowdown_ratio_tolerance", "slowdown_ratio_tolerance"),
        None,
    )
    consecutive_steps = _lookup_first(
        cfg,
        ("straggler_probe_consecutive_steps", "consecutive_steps"),
        5,
    )
    warmup_steps = _lookup_first(
        cfg,
        ("straggler_probe_warmup_steps", "warmup_steps"),
        10,
    )
    window = _lookup_first(
        cfg,
        ("straggler_probe_window", "straggler_probe_baseline_window", "window", "baseline_window"),
        20,
    )
    exclude = _lookup_first(
        cfg,
        ("straggler_probe_exclude", "exclude"),
        DEFAULT_EXCLUDE_REASONS,
    )
    exclude_after_ckpt = _lookup_first(
        cfg,
        ("straggler_probe_exclude_after_checkpoint_steps", "exclude_after_checkpoint_steps"),
        2,
    )
    cooldown_steps = _lookup_first(
        cfg,
        ("straggler_probe_cooldown_steps", "cooldown_steps"),
        50,
    )
    return cls(
        mode=mode,
        target_step_ms=target_step_ms,
        tolerance=tolerance,
        slowdown_ratio_tolerance=slowdown_tol,
        consecutive_steps=consecutive_steps,
        warmup_steps=warmup_steps,
        window=window,
        exclude=_normalize_exclude_tuple(exclude),
        exclude_after_checkpoint_steps=exclude_after_ckpt,
        cooldown_steps=cooldown_steps,
    )


@dataclasses.dataclass(frozen=True)
class TriggerDecision:
  """Per-step decision emitted by `StepTimeTrigger.observe`."""

  fired: bool
  mode: str
  cause: str
  streak_steps: list[int]
  reference_ms: float | None
  observed_ms: float | None
  excluded_steps: dict[int, str]
  step: int | None = None
  step_ms: float | None = None
  device_step_ms: float | None = None
  data_wait_ms: float = 0.0
  regression_ms: float | None = None
  target_step_ms: float | None = None
  baseline_median_ms: float | None = None
  recommended_stage: str = STAGE_NONE
  recommended_stages: tuple[str, ...] = ()
  in_cooldown: bool = False

  def to_summary_dict(self) -> dict[str, Any]:
    """Serializes the trigger decision into the dictionary format consumed by `report.py`."""
    return {
        "mode": self.mode,
        "fired": self.fired,
        "cause": self.cause,
        "step": self.step,
        "reference_ms": self.reference_ms,
        "target_step_ms": self.target_step_ms,
        "baseline_median_ms": self.baseline_median_ms,
        "observed_ms": self.observed_ms,
        "step_ms": self.step_ms,
        "device_step_ms": self.device_step_ms,
        "data_wait_ms": self.data_wait_ms,
        "regression_ms": self.regression_ms,
        "streak_steps": list(self.streak_steps),
        "excluded_steps": dict(self.excluded_steps),
        "recommended_stage": self.recommended_stage,
        "recommended_stages": list(self.recommended_stages),
        "in_cooldown": self.in_cooldown,
    }


@dataclasses.dataclass
class _ObservedStep:
  step: int
  step_ms: float
  data_wait_ms: float
  tags: tuple[str, ...]


class StepTimeTrigger:
  """Step-time anomaly detector with workload exclusions and data-wait subtraction."""

  def __init__(self, config: TriggerConfig | None = None, **kwargs: Any) -> None:
    if config is not None and kwargs:
      raise ValueError("Pass either a TriggerConfig instance or keyword arguments, not both")
    self.config: TriggerConfig = config if config is not None else TriggerConfig(**kwargs)
    self._history: list[_ObservedStep] = []
    self._explicit_marks: list[tuple[int, str]] = []
    self._excluded_steps: dict[int, str] = {}
    self._baseline_window: deque[float] = deque(maxlen=self.config.window)
    self._device_slow_streak: list[int] = []
    self._input_bound_streak: list[int] = []
    self._last_fire_step: int | None = None
    self._last_decision: TriggerDecision | None = None
    self._last_fired_decision: TriggerDecision | None = None

  def _is_reason_enabled(self, reason: str) -> bool:
    if reason in _CONFIGURABLE_EXCLUDE_REASONS:
      return reason in self.config.exclude
    return True

  def _apply_mark_to_excluded(self, step: int, reason: str) -> None:
    """Records `step` and any multi-step follow-ups in `_excluded_steps`."""
    norm_reason = str(reason).strip()
    if not norm_reason or not self._is_reason_enabled(norm_reason):
      return
    self._excluded_steps.setdefault(step, norm_reason)
    if norm_reason == "checkpoint":
      for offset in range(1, self.config.exclude_after_checkpoint_steps + 1):
        self._excluded_steps.setdefault(step + offset, "checkpoint")
    elif norm_reason == "eval":
      self._excluded_steps.setdefault(step + 1, "eval")
    elif norm_reason == "profiler":
      self._excluded_steps.setdefault(step + 1, "profiler")

  def mark(self, step: int, reason: str) -> None:
    """Marks `step` (and any follow-up steps for checkpoint/eval/profiler) as excluded.

    Can be called either before `observe(step, ...)` or after `observe(step, ...)`
    when an event (such as `maybe_save_checkpoint`, `eval`, or `maybe_deactivate_profiler`
    in MaxText's `train.py`) runs after `step_time_delta` is measured.

    Args:
      step: Training step number to mark.
      reason: Exclusion reason (`"profiler"`, `"checkpoint"`, `"eval"`,
        `"dropless_replay"`, `"recompile"`, `"warmup"`, or any custom tag).
    """
    step_idx = int(step)
    norm_reason = str(reason).strip()
    self._explicit_marks.append((step_idx, norm_reason))
    self._apply_mark_to_excluded(step_idx, norm_reason)
    if any(obs.step >= step_idx for obs in self._history):
      self._recompute_from_history()

  def _recompute_from_history(self) -> None:
    """Replays observed steps when `mark` is called retroactively on a past step."""
    saved_history = list(self._history)
    self._excluded_steps.clear()
    self._baseline_window.clear()
    self._device_slow_streak.clear()
    self._input_bound_streak.clear()
    self._last_fire_step = None
    self._last_decision = None
    self._last_fired_decision = None
    for marked_step, marked_reason in self._explicit_marks:
      self._apply_mark_to_excluded(marked_step, marked_reason)
    for item in saved_history:
      self._process_step(item.step, item.step_ms, item.data_wait_ms, item.tags)

  def _resolve_thresholds(self, baseline_median_ms: float | None) -> tuple[float | None, float | None, float | None]:
    """Returns `(target_threshold_ms, baseline_threshold_ms, default_reference_ms)`."""
    target_ms = self.config.target_step_ms
    target_thresh: float | None = None
    if self.config.mode in ("target", "both") and target_ms is not None:
      target_thresh = target_ms * (1.0 + self.config.tolerance)

    base_thresh: float | None = None
    if self.config.mode in ("baseline", "both") and baseline_median_ms is not None:
      base_thresh = baseline_median_ms * (1.0 + self.config.effective_slowdown_tolerance)

    if self.config.mode == "target":
      default_ref = target_ms
    elif self.config.mode == "baseline":
      default_ref = baseline_median_ms
    else:
      default_ref = target_ms if target_ms is not None else baseline_median_ms

    return target_thresh, base_thresh, default_ref

  def _check_exceeded(
      self,
      val_ms: float,
      target_thresh: float | None,
      base_thresh: float | None,
      baseline_median_ms: float | None,
      default_ref: float | None,
  ) -> tuple[bool, float | None]:
    """Checks whether `val_ms` exceeds the active threshold(s) and returns `(exceeded, ref_ms)`."""
    target_ms = self.config.target_step_ms
    if self.config.mode == "target":
      if target_thresh is not None and val_ms > target_thresh:
        return True, target_ms
      return False, target_ms
    if self.config.mode == "baseline":
      if base_thresh is not None and val_ms > base_thresh:
        return True, baseline_median_ms
      return False, baseline_median_ms

    # mode == "both"
    if target_thresh is not None and val_ms > target_thresh:
      return True, target_ms
    if base_thresh is not None and val_ms > base_thresh:
      return True, baseline_median_ms
    return False, default_ref

  def _process_step(
      self,
      step: int,
      step_ms: float,
      data_wait_ms: float,
      tags: tuple[str, ...],
  ) -> TriggerDecision:
    """Evaluates a single observed step and updates streak/baseline state."""
    for tag in tags:
      self._apply_mark_to_excluded(step, tag)

    if 0 <= step < self.config.warmup_steps:
      self._excluded_steps.setdefault(step, "warmup")

    clean_data_wait_ms = max(0.0, float(data_wait_ms))
    if "data_wait" not in self.config.exclude:
      clean_data_wait_ms = 0.0
    device_step_ms = max(0.0, float(step_ms) - clean_data_wait_ms)

    baseline_median_ms = round(float(statistics.median(self._baseline_window)), 3) if self._baseline_window else None
    target_thresh, base_thresh, default_ref = self._resolve_thresholds(baseline_median_ms)

    in_cooldown = (
        self._last_fire_step is not None
        and self.config.cooldown_steps > 0
        and 0 < (step - self._last_fire_step) <= self.config.cooldown_steps
    )

    # Excluded steps never count toward a streak, never reset an active streak,
    # and are never added to the clean baseline window.
    if step in self._excluded_steps:
      active_streak = list(self._device_slow_streak or self._input_bound_streak)
      ref_ms = round(default_ref, 3) if default_ref is not None else None
      obs_ms = round(device_step_ms, 3)
      reg_ms = round(obs_ms - ref_ms, 3) if ref_ms is not None else None
      decision = TriggerDecision(
          fired=False,
          mode=self.config.mode,
          cause=CAUSE_NONE,
          streak_steps=active_streak,
          reference_ms=ref_ms,
          observed_ms=obs_ms,
          excluded_steps=dict(self._excluded_steps),
          step=step,
          step_ms=round(float(step_ms), 3),
          device_step_ms=obs_ms,
          data_wait_ms=round(clean_data_wait_ms, 3),
          regression_ms=reg_ms,
          target_step_ms=self.config.target_step_ms,
          baseline_median_ms=baseline_median_ms,
          recommended_stage=STAGE_NONE,
          recommended_stages=(),
          in_cooldown=in_cooldown,
      )
      self._last_decision = decision
      return decision

    device_exceeded, dev_ref = self._check_exceeded(
        device_step_ms, target_thresh, base_thresh, baseline_median_ms, default_ref
    )
    total_exceeded, tot_ref = self._check_exceeded(
        float(step_ms), target_thresh, base_thresh, baseline_median_ms, default_ref
    )

    if not device_exceeded and not total_exceeded:
      self._baseline_window.append(device_step_ms)
      baseline_median_ms = round(float(statistics.median(self._baseline_window)), 3)
      if self.config.mode == "baseline" or (self.config.mode == "both" and self.config.target_step_ms is None):
        default_ref = baseline_median_ms

    if in_cooldown:
      self._device_slow_streak.clear()
      self._input_bound_streak.clear()
      ref_ms = round(default_ref, 3) if default_ref is not None else None
      obs_ms = round(device_step_ms, 3)
      reg_ms = round(obs_ms - ref_ms, 3) if ref_ms is not None else None
      decision = TriggerDecision(
          fired=False,
          mode=self.config.mode,
          cause=CAUSE_NONE,
          streak_steps=[],
          reference_ms=ref_ms,
          observed_ms=obs_ms,
          excluded_steps=dict(self._excluded_steps),
          step=step,
          step_ms=round(float(step_ms), 3),
          device_step_ms=obs_ms,
          data_wait_ms=round(clean_data_wait_ms, 3),
          regression_ms=reg_ms,
          target_step_ms=self.config.target_step_ms,
          baseline_median_ms=baseline_median_ms,
          recommended_stage=STAGE_NONE,
          recommended_stages=(),
          in_cooldown=True,
      )
      self._last_decision = decision
      return decision

    fired = False
    cause = CAUSE_NONE
    recommended_stage = STAGE_NONE
    recommended_stages: tuple[str, ...] = ()
    active_ref = default_ref

    if device_exceeded:
      self._device_slow_streak.append(step)
      self._input_bound_streak.clear()
      active_ref = dev_ref
      streak_steps = list(self._device_slow_streak)
      if len(self._device_slow_streak) >= self.config.consecutive_steps:
        fired = True
        cause = CAUSE_DEVICE_SLOW
        recommended_stage = STAGE_A_HOST_TIMING
        recommended_stages = ALL_PROBE_STAGES
        self._last_fire_step = step
        self._device_slow_streak.clear()
    elif total_exceeded:
      # device_step_ms is within threshold, but step_ms exceeds threshold:
      # data_wait_ms explains the excess. Keep healthy device_step_ms in baseline window.
      self._baseline_window.append(device_step_ms)
      self._input_bound_streak.append(step)
      self._device_slow_streak.clear()
      active_ref = tot_ref
      streak_steps = list(self._input_bound_streak)
      if len(self._input_bound_streak) >= self.config.consecutive_steps:
        fired = True
        cause = CAUSE_INPUT_BOUND
        recommended_stage = STAGE_A_HOST_TIMING
        recommended_stages = (STAGE_A_HOST_TIMING,)
        self._last_fire_step = step
        self._input_bound_streak.clear()
    else:
      self._device_slow_streak.clear()
      self._input_bound_streak.clear()
      streak_steps = []

    ref_ms = round(active_ref, 3) if active_ref is not None else None
    obs_ms = round(device_step_ms, 3)
    reg_ms = round(obs_ms - ref_ms, 3) if ref_ms is not None else None

    decision = TriggerDecision(
        fired=fired,
        mode=self.config.mode,
        cause=cause,
        streak_steps=streak_steps,
        reference_ms=ref_ms,
        observed_ms=obs_ms,
        excluded_steps=dict(self._excluded_steps),
        step=step,
        step_ms=round(float(step_ms), 3),
        device_step_ms=obs_ms,
        data_wait_ms=round(clean_data_wait_ms, 3),
        regression_ms=reg_ms,
        target_step_ms=self.config.target_step_ms,
        baseline_median_ms=baseline_median_ms,
        recommended_stage=recommended_stage,
        recommended_stages=recommended_stages,
        in_cooldown=False,
    )
    self._last_decision = decision
    if fired:
      self._last_fired_decision = decision
    return decision

  def observe(
      self,
      step: int,
      step_ms: float,
      data_wait_ms: float = 0.0,
      tags: Iterable[str] | str = (),
  ) -> TriggerDecision:
    """Observes a training step's timing and returns a `TriggerDecision`.

    Args:
      step: 0-based training step index.
      step_ms: Total wall-clock step time in milliseconds (`step_time_delta`).
      data_wait_ms: Host input pipeline wait time in milliseconds subtracted to
        compute `device_step_ms = step_ms - data_wait_ms`.
      tags: Optional event tags on this step (`"profiler"`, `"checkpoint"`,
        `"eval"`, `"dropless_replay"`, `"recompile"`, `"warmup"`, or custom).

    Returns:
      A `TriggerDecision` describing whether the trigger fired, the cause
      (`DEVICE_SLOW`, `INPUT_BOUND`, or `none`), streak steps, reference/observed
      timings, and excluded steps with reasons.
    """
    if isinstance(tags, str):
      norm_tags = (tags.strip(),) if tags.strip() else ()
    else:
      norm_tags = tuple(str(t).strip() for t in tags if str(t).strip())
    record = _ObservedStep(
        step=int(step),
        step_ms=float(step_ms),
        data_wait_ms=float(data_wait_ms),
        tags=norm_tags,
    )
    self._history.append(record)
    return self._process_step(record.step, record.step_ms, record.data_wait_ms, record.tags)

  @property
  def excluded_steps(self) -> dict[int, str]:
    """Returns a copy of all currently excluded steps mapped to their reason."""
    return dict(self._excluded_steps)

  @property
  def baseline_median_ms(self) -> float | None:
    """Returns the rolling median of clean `device_step_ms` values, or `None` if empty."""
    if not self._baseline_window:
      return None
    return round(float(statistics.median(self._baseline_window)), 3)

  def summary(self) -> dict[str, Any]:
    """Returns a summary dictionary suitable for `report.build_straggler_report`."""
    chosen = self._last_fired_decision or self._last_decision
    if chosen is not None:
      data = chosen.to_summary_dict()
      data["excluded_steps"] = dict(self._excluded_steps)
      data["baseline_median_ms"] = self.baseline_median_ms
      return data
    return {
        "mode": self.config.mode,
        "fired": False,
        "cause": CAUSE_NONE,
        "step": None,
        "reference_ms": self.config.target_step_ms,
        "target_step_ms": self.config.target_step_ms,
        "baseline_median_ms": None,
        "observed_ms": None,
        "step_ms": None,
        "device_step_ms": None,
        "data_wait_ms": 0.0,
        "regression_ms": None,
        "streak_steps": [],
        "excluded_steps": dict(self._excluded_steps),
        "recommended_stage": STAGE_NONE,
        "recommended_stages": [],
        "in_cooldown": False,
    }


Trigger = StepTimeTrigger

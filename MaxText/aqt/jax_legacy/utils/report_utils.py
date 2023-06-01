# Copyright 2022 Google LLC
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

"""Util functions to generate an experiment report after training.

Please refer to the README.md for an overview of the reporting tool.
"""

import dataclasses
import enum
import functools
import json
import os
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from aqt.jax_legacy.utils import hparams_utils
from aqt.jax_legacy.utils import tfevent_utils
import numpy as onp
from tensorflow.io import gfile

# Constants
COMPUTE_MEMORY_COST_FILENAME = 'compute_memory_cost.json'
HPARAMS_CONFIG_FILENAME = 'hparams_config.json'
REPORT_FILENAME = 'report.json'

EventSeries = tfevent_utils.EventSeries

# Type Aliases
# Nested dict mapping from component (first key), attribute (second key) to
# events stored in EventSeries. E.g. component = 'train', attribute = 'loss'.
_AllEvents = Dict[str, Dict[str, EventSeries]]

# Nested dict mapping from component (first key), attribute (second key) to
# aggregated metric (float). E.g. component = 'train', attribute = 'loss'.
_AllAggMetrics = Dict[str, Dict[str, float]]


@enum.unique
class MinOrMax(enum.Enum):
  """Aggregation function to use for finding early stopping step."""
  MIN = enum.auto()  # use min value for early stopping step.
  MAX = enum.auto()  # use max value for early stopping step.

  def get_func(self) -> Callable[[onp.ndarray], int]:
    """Returns function associated with enum option. See parent class."""
    if self == MinOrMax.MIN:
      return onp.nanargmin
    elif self == MinOrMax.MAX:
      return onp.nanargmax
    else:
      raise ValueError('MinOrMax enum option not recognized.')


@enum.unique
class SmoothingKernel(enum.Enum):
  """Kernel function to use for smoothing."""
  # RECTANGULAR:Every value in symmetric window weighted equally. Values
  # outside the window are not included in average.
  # TRIANGULAR: Every value in symmetric window weighted as a linear function of
  # absolute distance to kernel center. Values outside the window are not
  # included in average.
  RECTANGULAR = enum.auto()
  TRIANGULAR = enum.auto()

  def rectangular_kernel(self, x: float, window_size_in_steps: int) -> float:
    """Rectangular kernel for moving window average.

    All values in window are equally weighted.

    Args:
      x: Distance to kernel center in steps.
      window_size_in_steps: Size of the window to average over.

    Returns:
      Unnormalized weight to use for averaging, e.g. in `np.average()`.

    Raises:
      ValueError: If window_size_in_steps arg is less than 1.
    """
    if window_size_in_steps < 1:
      raise ValueError('window_size_in_steps has to be >= 1.')
    if abs(x) <= window_size_in_steps / 2:
      return 1.0
    else:
      return 0.0

  def triangular_kernel(self, x: float, window_size_in_steps: int) -> float:
    """Triangular kernel for moving window average.

    The weight is a linear function of the absolute distance to the kernel
    center.

    Args:
      x: Distance to kernel center in steps.
      window_size_in_steps: Size of the window to average over.

    Returns:
      Unnormalized weight to use for averaging, e.g. in `np.average()`.

    Raises:
      ValueError: If window_size_in_steps arg is less than 1.
    """
    if window_size_in_steps < 1:
      raise ValueError('window_size_in_steps has to be >= 1.')
    return max(0.0, window_size_in_steps / 2 - abs(x))

  def get_func(
      self,
      window_size_in_steps: Optional[int] = None) -> Callable[[float], float]:
    """Returns function associated with enum option. See parent class."""
    if self == SmoothingKernel.RECTANGULAR:
      if window_size_in_steps is None:
        raise ValueError('For rectangular smoothing_kernel '
                         'window_size_in_steps must be provided.')
      return functools.partial(
          self.rectangular_kernel, window_size_in_steps=window_size_in_steps)
    elif self == SmoothingKernel.TRIANGULAR:
      if window_size_in_steps is None:
        raise ValueError('For triangular smoothing_kernel '
                         'window_size_in_steps must be provided.')
      return functools.partial(
          self.triangular_kernel, window_size_in_steps=window_size_in_steps)
    else:
      raise ValueError('SmoothingKernel enum option not recognized.')




@dataclasses.dataclass
class ExperimentReport:
  """Report for a single experiment run based on its TFEvents files."""
  # Model directory corresponding to single run, with TFEvents files to
  # generate report from.
  model_dir: str

  # Metrics at early stop step, with smoothing applied.
  # If NaN values present, then this field will
  # be left None, but unsmoothed_metrics will still be reported.
  # maps component name (e.g. eval) to metrics dict, which in turn maps
  # attribute name to scalar value.
  metrics: Optional[_AllAggMetrics]

  # Metrics without smoothing at early stop step.
  # maps component name (e.g. eval) to metrics dict, which in turn maps
  # attribute name to scalar value.
  unsmoothed_metrics: Optional[_AllAggMetrics]

  # Step at which early_stop_attr in early_stop_ds_dir is minimized. Scalars are
  # reported at this step.
  early_stop_step: int

  # Number of training steps. In combination with early_stop_step, can help
  # determine whether training converged and started to overfit.
  num_train_steps: int

  # Arguments passed into create_end_of_training_report(), the function that
  # created this report.
  # Included here because the arguments can impact the reported metrics, e.g.
  # which attribute was used to find the early stopping step.
  report_query_args: Dict[str, Any]

  # Human-readable experiment name.
  experiment_name: Optional[str] = None

  # Name of user who launched the experiment.
  user_name: Optional[str] = None

  # When experiment was launched, formatted as '%Y%m%dT%H%M%S'.
  launch_time: Optional[str] = None

  # Evaluation frequency. How often summaries were saved to file.
  eval_freq: Optional[int] = None

  # If any metrics contain NaN values, first step at which a NaN value occurs.
  first_nan_step: Optional[int] = None

  # Tensorboard ID or URL.
  tensorboard_id: Optional[str] = None

  # Full path to hparams config file.
  hparams_config_path: Optional[str] = None



def check_for_nans(event_series: EventSeries, start_step: int) -> Optional[int]:
  """Finds step >= start_step at which first NaN value occurs if there are any.

  Args:
    event_series: list of tuples (step, value).
    start_step: After which step to check for NaNs.

  Returns:
    Step at which first NaN value occurs, or None otherwise.

  """
  keep_indices = (event_series.steps >= start_step)
  event_series.steps = event_series.steps[keep_indices]
  event_series.values = event_series.values[keep_indices]
  nan_indices = onp.argwhere(onp.isnan(event_series.values))
  if nan_indices.size:
    return int(event_series.steps[onp.min(nan_indices)])
  return None


def check_all_events_for_nans(all_events: _AllEvents) -> Optional[int]:
  """Finds step at which first NaN value occurs if there are any.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.

  Returns:
    Step at which first NaN value occurs, or None otherwise.
  """

  first_nan_step = None

  for events_dict in all_events.values():
    for events in events_dict.values():
      cur_first_nan_step = check_for_nans(events, start_step=0)
      if cur_first_nan_step is None:
        continue
      if first_nan_step is None:
        first_nan_step = cur_first_nan_step
      else:
        first_nan_step = min(first_nan_step, cur_first_nan_step)
  return first_nan_step


def find_early_stop_step(event_series: EventSeries,
                         early_stop_func: Callable[[onp.ndarray], int],
                         start_step: int) -> int:
  """Finds step >= start_step at which event_series is minimized.

  Args:
    event_series: list of tuples (step, value).
    early_stop_func: Aggregator function to use to find early_stop_step.
    start_step: After which step to include values in moving average.

  Returns:
    Step at which moving average of series is minimized.

  """
  keep_indices = (event_series.steps >= start_step)
  event_series.steps = event_series.steps[keep_indices]
  event_series.values = event_series.values[keep_indices]

  if event_series.steps.size == 0:
    raise ValueError('event_series does not have events after start_step.')

  if onp.all(onp.isnan(event_series.values)):
    return start_step
  early_stop_idx = early_stop_func(event_series.values)
  return int(event_series.steps[early_stop_idx])


def apply_smoothing_about_step(events: EventSeries, step: int,
                               kernel_fn: Callable[[float], float]) -> float:
  """Applies smoothing of event values for a single step.

  Args:
    events: list of tuples (step, value).
    step: Step to apply smoothing about.
    kernel_fn: Kernel function to use for smoothing.

  Returns:
    Smoothed value at step.

  Raises:
    ValueError: If NaN values present in events.values.
  """
  if check_for_nans(events, start_step=0) is not None:
    raise ValueError(
        'NaN values encountered in smoothing, which is not supported.')
  weights = onp.vectorize(kernel_fn)(events.steps - step)
  return float(onp.average(events.values, weights=weights))


def apply_smoothing(events: EventSeries,
                    kernel_fn: Callable[[float], float]) -> EventSeries:
  """Applies smoothing of event values over all steps.

  Args:
    events: list of tuples (step, value).
    kernel_fn: Kernel function to use for smoothing.

  Returns:
    Smoothed events for all steps in steps arg.
  """
  smoothed_events = EventSeries(
      name=events.name,
      steps=onp.array([], dtype=int),
      values=onp.array([]),
      wall_times=None)
  for i in range(len(events.steps)):
    smoothed_events.steps = onp.append(smoothed_events.steps, events.steps[i])
    smoothed_value = apply_smoothing_about_step(
        events=events, step=events.steps[i], kernel_fn=kernel_fn)
    smoothed_events.values = onp.append(smoothed_events.values, smoothed_value)

  return smoothed_events


def get_agg_metrics_at_step(
    all_events: _AllEvents, step: int,
    smoothing_kernel_fn: Optional[Callable[[float], float]]) -> _AllAggMetrics:
  """Computes aggregated metrics from EventSeries dicts at early stop step.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.
    step: Step at which to get event values to compute aggregated metrics.
    smoothing_kernel_fn: If None, no smoothing will be applied. If any NaNs are
      present, has to be set to None, otherwise ValueError will be raised.

  Returns:
    dict mapping from (component, attribute) to aggregated scalar metric.

  """
  all_agg_metrics = {}
  for component, events_dict in all_events.items():
    agg_metrics_dict = {}
    for attr, events in events_dict.items():
      if smoothing_kernel_fn is None:
        index = onp.argmin(onp.abs(events.steps - step))
        agg_metrics_dict[attr] = events.values[index]
      else:
        agg_metrics_dict[attr] = apply_smoothing_about_step(
            events, step=step, kernel_fn=smoothing_kernel_fn)
    all_agg_metrics[str(component)] = agg_metrics_dict

  return all_agg_metrics


def compute_agg_metrics_from_events(
    all_events: _AllEvents,
    early_stop_component: str,
    early_stop_attr: str,
    early_stop_agg: MinOrMax,
    smoothing_kernel: SmoothingKernel,
    window_size_in_steps: Optional[int] = None,
    start_step: int = 0
) -> Tuple[_AllAggMetrics, Optional[_AllAggMetrics], int, Optional[int]]:
  """Computes aggregated metrics from EventSeries dicts.

  Args:
    all_events: Nested dict mapping from component, attribute, to EventSeries.
    early_stop_component: Which component to use to find early_stop_step.
    early_stop_attr: Attribute to find minimum or maximum of, e.g. 'perplexity'.
    early_stop_agg: Which aggregator to use to find early_stop_step. See
      MinOrMax class for enum options.
    smoothing_kernel: Which kernel to use for smoothing. See SmoothingKernel
      class for enum options.
    window_size_in_steps: Only applicable to some kernels, including
      'rectangular' kernel. Number of steps to average over.
    start_step: After which step to consider early stopping, e.g. if set to 100,
      only steps >= 100 will be considered.

  Returns:
    Tuple of dict mapping from (component, attribute) to aggregated scalar
      metric and early_stop_step.

  """
  first_nan_step = check_all_events_for_nans(all_events=all_events)

  early_stop_func = early_stop_agg.get_func()
  early_stop_events = all_events[early_stop_component][early_stop_attr]

  if first_nan_step is None:
    # Apply smoothing to early stop component events.
    smoothing_kernel_func = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    early_stop_events = apply_smoothing(
        events=early_stop_events, kernel_fn=smoothing_kernel_func)

  early_stop_step = find_early_stop_step(
      early_stop_events, early_stop_func=early_stop_func, start_step=start_step)

  all_agg_metrics_unsmoothed = get_agg_metrics_at_step(
      all_events=all_events, step=early_stop_step, smoothing_kernel_fn=None)

  if first_nan_step is None:
    # Only get smoothed metrics if no NaN values found.
    all_metrics_smoothed = get_agg_metrics_at_step(
        all_events=all_events,
        step=early_stop_step,
        smoothing_kernel_fn=smoothing_kernel_func)
  else:
    all_metrics_smoothed = None

  return all_agg_metrics_unsmoothed, all_metrics_smoothed, early_stop_step, first_nan_step


def create_end_of_training_report(
    model_dir: str,
    eval_freq: int,
    num_train_steps: int,
    early_stop_attr: str,
    early_stop_agg: MinOrMax,
    smoothing_kernel: SmoothingKernel,
    early_stop_ds_dir: Optional[str] = None,
    other_ds_dirs: Optional[List[str]] = None,
    tags_to_include: Optional[List[str]] = None,
    window_size_in_steps: int = 1,
    start_step: int = 0,
    experiment_name: Optional[str] = None,
    user_name: Optional[str] = None,
    launch_time: Optional[str] = None,
    tensorboard_id: Optional[str] = None,
) -> ExperimentReport:
  """Creates an experiment report from TFEvents data after training completion.

  Args:
    model_dir: A model directory corresponding to a single model run, with
      TFEvent file(s) and a single hparams_config file. The TFEvent files can
      either be stored directly in model_dir, or in subdirectories in model_dir,
      but not both.
    eval_freq: Frequency of event saving.
    num_train_steps: Number of training steps.
    early_stop_attr: Attribute to find minimum or maximum of, e.g. 'perplexity'.
    early_stop_agg: Which aggregator to use to find early_stop_step. See
      MinOrMax class for enum options.
    smoothing_kernel: Which kernel to use for smoothing. See SmoothingKernel
      class for enum options.
    early_stop_ds_dir: The events subdir in model_dir to use to find
      early_stop_step if model_dir has subdirs. The early_stop_attr within
      early_stop_ds_dir will be used to find the early_stop_step.
    other_ds_dirs: List of other subdirs in model_dir with events to report.
    tags_to_include: List of event tags that should be included.
    window_size_in_steps: Number of steps to average over. Should be multiple of
      eval_freq. If set to 1, no averaging will be applied.
    start_step: After which step to consider early stopping, e.g. if set to 100,
      only steps >= 100 will be considered.
    experiment_name:  Human-readable experiment name.
    user_name: Name of user who launched the experiment.
    launch_time: When experiment was launched, formatted as '%Y%m%dT%H%M%S'.
    tensorboard_id: Tensorboard ID, e.g. URL to tensorboard dev, if applicable.

  Returns:
    An ExperimentReport dataclass instance.

  """
  # Saving report query args, to be included in the report.
  report_query_args = {
      'early_stop_attr': early_stop_attr,
      'early_stop_agg': early_stop_agg.name,
      'early_stop_ds_dir': early_stop_ds_dir,
      'other_ds_dirs': other_ds_dirs,
      'tags_to_include': tags_to_include,
      'smoothing_kernel': smoothing_kernel.name,
      'window_size_in_steps': window_size_in_steps,
      'start_step': start_step,
  }

  all_events = {}

  early_stop_component = None

  # If subdirs provided
  if early_stop_ds_dir is not None or other_ds_dirs is not None:
    if early_stop_ds_dir is None:
      raise ValueError(
          'If other_ds_dirs is not None, early_stop_ds_dir has to be '
          'provided.')

    early_stop_events = tfevent_utils.get_parsed_tfevents(
        os.path.join(model_dir, early_stop_ds_dir), tags_to_include)
    early_stop_component = early_stop_ds_dir
    all_events[early_stop_component] = early_stop_events

    if other_ds_dirs is not None:
      for ds_dir in other_ds_dirs:
        if ds_dir is not None:
          all_events[ds_dir] = tfevent_utils.get_parsed_tfevents(
              os.path.join(model_dir, ds_dir), tags_to_include)
  else:
    # If no subdirs provided, will assume that there are no subcomponents.
    # For consistency with the case when we do have components, we store the
    # events under the dummy component 'all'.
    early_stop_component = 'all'
    all_events[early_stop_component] = tfevent_utils.get_parsed_tfevents(
        model_dir, tags_to_include)

  all_agg_metrics_unsmoothed, all_agg_metrics_smoothed, early_stop_step, first_nan_step = compute_agg_metrics_from_events(
      all_events=all_events,
      early_stop_component=early_stop_component,
      early_stop_attr=early_stop_attr,
      early_stop_agg=early_stop_agg,
      smoothing_kernel=smoothing_kernel,
      window_size_in_steps=window_size_in_steps,
      start_step=start_step)

  report = ExperimentReport(
      model_dir=model_dir,
      metrics=all_agg_metrics_smoothed,
      unsmoothed_metrics=all_agg_metrics_unsmoothed,
      early_stop_step=early_stop_step,
      num_train_steps=num_train_steps,
      eval_freq=eval_freq,
      first_nan_step=first_nan_step,
      experiment_name=experiment_name,
      user_name=user_name,
      launch_time=launch_time,
      tensorboard_id=tensorboard_id,
      report_query_args=report_query_args,
  )

  return report


def report_path_from_model_dir(model_dir: pathlib.Path) -> pathlib.Path:
  """Generates report path from model_dir."""
  report_path = '--'.join(list(model_dir.parts)[1:])

  # get rid of file extensions if there were any
  report_path = os.path.splitext(str(report_path))[0]

  return pathlib.Path(report_path)


def save_report(report: ExperimentReport,
                report_dir: pathlib.Path) -> pathlib.Path:
  """Saves json serialized ExperimentReport instance to report directory.

  If report has hparams_config_path, will attempt to copy over the config file
  to the report_dir as well.

  The filename is auto-generated by concatenating report.model_dir.

  Args:
    report: The report to be saved.
    report_dir: The directory to save to.

  Returns:
    Path where report was saved to.
  """
  report_path = report_path_from_model_dir(pathlib.Path(report.model_dir))
  if not gfile.exists(report_dir / report_path):
    gfile.makedirs(report_dir / report_path)
  if report.hparams_config_path is not None:
    gfile.copy(
        report.hparams_config_path,
        report_dir / report_path / HPARAMS_CONFIG_FILENAME,
        overwrite=True)
    report.hparams_config_path = str(report_dir / report_path /
                                     HPARAMS_CONFIG_FILENAME)
  data_dict = dataclasses.asdict(report)
  path = pathlib.Path(report_dir / report_path / REPORT_FILENAME)
  with gfile.GFile(path, 'w') as file:
    json.dump(data_dict, file, indent=2)
  return report_dir / report_path



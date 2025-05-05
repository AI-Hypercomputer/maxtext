"""Progress bar."""

from absl import logging
from tqdm import auto
from tunix.sft import metrics_logger as ml

tqdm = auto.tqdm


class ProgressBar:
  """Progress bar."""

  def __init__(
      self,
      metrics_logger: ml.MetricsLogger,
      initial_steps: int,
      max_steps: int,
  ):

    # Initialise progress bar.
    self.tqdm_bar = tqdm(
        total=max_steps,
        initial=initial_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        leave=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
            "{rate_fmt}{postfix}]"
        ),
    )

    # Also, initialise a dictionary for metrics.
    self.metrics = {}
    self.initial_steps = initial_steps
    self.max_steps = max_steps
    self.metrics_logger = metrics_logger

  def _update_metric(self, metric_name: str, mode: ml.Mode, ndigits: int = 3):
    """Update metric corresponding to `metric_name`."""

    mode_str = str(mode)
    if not self.metrics_logger.metric_exists(metric_name, mode):
      logging.warning(
          "Metric %s not found for mode %s. Not logging metric.",
          metric_name,
          mode_str,
      )
      return

    self.metrics[f"{mode_str}_{metric_name}"] = round(
        self.metrics_logger.get_metric(metric_name, mode).item(),
        ndigits,
    )

  def update_metrics(
      self, metric_names: list[str], mode: ml.Mode, ndigits: int = 3
  ):
    """Update metrics corresponding to `metric_names`."""

    for metric_name in metric_names:
      self._update_metric(metric_name, mode, ndigits)

  def _sort_metrics(self):
    """Sort metrics by mode."""
    sorted_metrics = {}

    for mode_metric_name, metric_value in self.metrics.items():
      mode = mode_metric_name.split("_")[0]
      if mode == "train":
        sorted_metrics[mode_metric_name] = metric_value

    for mode_metric_name, metric_value in self.metrics.items():
      mode = mode_metric_name.split("_")[0]
      if mode == "eval":
        sorted_metrics[mode_metric_name] = metric_value

    self.metrics = sorted_metrics

  def update(self, n: int = 1):
    """Update progress bar."""
    if self.metrics:
      self._sort_metrics()
      self.tqdm_bar.set_postfix(self.metrics)
    self.tqdm_bar.update(n)

  def close(self):
    del self.metrics
    self.tqdm_bar.close()

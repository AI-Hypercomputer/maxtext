# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run-owned scalar backends for held-out RL evaluation.

Tunix closes its actor logger when train() returns. These backends live until
the final MaxText evaluation finishes. Calling them directly avoids registering
another global JAX listener (which would duplicate all actor metrics).
"""

import time

import jax
from tensorboardX import SummaryWriter


class _TensorboardBackend:
  """A unique event filename alongside Tunix's event file in the same run."""

  def __init__(self, config):
    self._writer = SummaryWriter(logdir=config.tensorboard_dir, filename_suffix=".maxtext_rl")
    self._flush_period = config.log_period

  def log_scalar(self, name, value, *, step):
    self._writer.add_scalar(name, value, step)
    if step % self._flush_period == 0:
      self._writer.flush()

  def close(self):
    self._writer.close()


class RLMetricLogger:
  """Writes MaxText metrics on process zero and flushes them at run shutdown."""

  def __init__(self, config):
    self._backends = []
    if jax.process_index() != 0:
      return
    if getattr(config, "enable_tensorboard", False):
      self._backends.append(_TensorboardBackend(config))
    if getattr(config, "managed_mldiagnostics", False):
      from tunix.rl.metrics.mldiag_backend import MLDiagScalarBackend  # pylint: disable=import-outside-toplevel

      self._backends.append(MLDiagScalarBackend())

  def write(self, values, step):
    """Writes scalars at a completed rollout-policy (outer) step."""
    for name, value in values.items():
      for backend in self._backends:
        backend.log_scalar(name, float(value), step=int(step))

  def close(self):
    for backend in self._backends:
      backend.close()
    self._backends.clear()

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()


def log_evaluation(logger, cluster, result, metadata, duration, config):
  """Exports held-out scores with explicit units, denominators and step axes."""
  if logger is None:
    return
  _, total, accuracy, partial_accuracy, format_accuracy, mean_reward = result
  values = {
      "rl/eval/accuracy_percent": accuracy,
      "rl/eval/partial_accuracy_percent": partial_accuracy,
      "rl/eval/format_accuracy_percent": format_accuracy,
      "rl/eval/num_prompts": total,
      "rl/eval/num_samples": metadata["num_samples"],
      "rl/eval/num_reward_samples": metadata["num_reward_samples"],
      "rl/eval/duration_seconds": duration,
      "rl/eval/success": 1,
      "rl/eval/num_passes": config.num_eval_passes,
      "rl/steps/actor_updates": cluster.actor_trainer.train_steps,
      "rl/steps/outer": cluster.global_steps,
  }
  # Empty evaluations must not manufacture measured zero accuracy/reward.
  if not total or not metadata["num_samples"]:
    for name in ("accuracy_percent", "partial_accuracy_percent", "format_accuracy_percent"):
      del values[f"rl/eval/{name}"]
    values["rl/eval/success"] = 0
  if metadata["num_reward_samples"]:
    values["rl/eval/mean_reward"] = mean_reward
  logger.write(values, cluster.global_steps)


def log_evaluation_failure(logger, cluster, duration):
  """Records failure without manufacturing zero-valued accuracy/reward."""
  if logger is not None:
    logger.write({"rl/eval/success": 0, "rl/eval/duration_seconds": duration}, cluster.global_steps)


def evaluate_with_logging(evaluate_fn, config, dataset, cluster, **kwargs):
  """Runs any evaluation stage using the run-owned backends and outer step."""
  logger = getattr(cluster, "maxtext_metrics_logger", None)
  started = time.perf_counter()
  metadata = {}
  try:
    result, responses = evaluate_fn(config, dataset, rl_cluster=cluster, metrics=metadata, **kwargs)
  except Exception:
    log_evaluation_failure(logger, cluster, time.perf_counter() - started)
    raise
  log_evaluation(logger, cluster, result, metadata, time.perf_counter() - started, config)
  return result, responses

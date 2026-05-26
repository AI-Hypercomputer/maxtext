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

"""Training hooks for post-train RL."""

from typing import Any, Callable, Optional

from tunix.sft import hooks as _tunix_hooks

from maxtext.trainers.post_train.rl.evaluate_rl import evaluate
from maxtext.utils import max_logging


class RLTrainingHooks(_tunix_hooks.TrainingHooks):
  """Tunix `TrainingHooks` subclass that fires `evaluate(...)` every
  `eval_interval` outer steps during RL training.

  tunix's `eval_every_n_steps` in `RLTrainingConfig` is silently dead unless
  an `eval_ds` is passed to `trainer.train()`, and even then tunix's default
  `_run_eval` re-runs the full GRPO rollout (`num_generations` sampled per
  prompt), which is ~3hr/eval and impractical for trajectory monitoring.

  This hook hooks `on_train_step_end`, checks
  `rl_cluster.global_steps % eval_interval`, and calls maxtext's
  `evaluate(...)` (using whichever `eval_sampling_strategy` is configured
  in `generation_configs`) plus the configured scoring pipeline,
  logging the result. Gives matched-step PRE/INTERMEDIATE/POST curves
  without any change to tunix.
  """

  def __init__(
      self,
      rl_cluster: Any,
      trainer_config: Any,
      test_dataset: Any,
      eval_interval: int,
      reward_fns: Optional[list[Callable[..., Any]]] = None,
  ):
    self._rl_cluster = rl_cluster
    self._trainer_config = trainer_config
    self._test_dataset = test_dataset
    self._eval_interval = eval_interval
    self._reward_fns = reward_fns
    self._last_step_evaluated = -1

  # The five lifecycle methods below are abstract in `tunix.sft.hooks.TrainingHooks`,
  # so subclasses MUST implement them. We have no per-step / per-train work to do here
  # outside `on_train_step_end`, so they're no-op stubs.
  def on_train_start(self, train_ctx):  # noqa: ARG002
    del train_ctx

  def on_train_end(self, train_ctx):  # noqa: ARG002
    del train_ctx

  def on_train_step_start(self, train_ctx):  # noqa: ARG002
    del train_ctx

  def on_eval_step_start(self, train_ctx):  # noqa: ARG002
    del train_ctx

  def on_eval_step_end(self, train_ctx, *args, **kwargs):  # noqa: ARG002
    del train_ctx, args, kwargs

  def on_train_step_end(self, trainer, step, loss):  # noqa: ARG002
    """Fire `evaluate(...)` once per `eval_interval` outer steps."""
    del trainer, loss
    try:
      outer_step = int(self._rl_cluster.global_steps)
    except Exception:  # pylint: disable=broad-exception-caught
      outer_step = int(step) if step is not None else -1
    if outer_step <= 0 or outer_step == self._last_step_evaluated:
      return
    if outer_step % self._eval_interval != 0:
      return
    self._last_step_evaluated = outer_step
    try:
      tc = self._trainer_config
      (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = evaluate(
          tc,
          self._test_dataset,
          rl_cluster=self._rl_cluster,
          num_passes=tc.num_eval_passes,
          corr_lst=tc.eval_corr_lst,
          make_lst=tc.eval_make_lst,
          reward_fns=self._reward_fns,
      )
      max_logging.warning(
          f"Intermediate Eval (step={outer_step}): {corr=}, {total=},"
          f" {accuracy=}%, {partial_accuracy=}%, {format_accuracy=}%,"
          f" {mean_reward=:.4f}"
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.warning(f"[intermediate-eval] step={outer_step} failed: {e!r}")

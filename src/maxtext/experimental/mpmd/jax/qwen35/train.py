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

"""Loss and the single-controller train step for Qwen3.5.

Everything model-agnostic is the DeepSeek port's (`src/maxtext/experimental/mpmd/jax/deepseek/train.py`,
MaxText's trainer reproduced in raw JAX): the T5X custom-VJP cross entropy,
AdamW, global-norm clipping, the LR schedule and the synthetic batch.  They are
re-exported here so a Qwen caller never has to reach into the DeepSeek package.
Only the model call differs: Qwen3.5 has no router bias, hence no `aux` tree.
"""

from __future__ import annotations

import jax

from maxtext.experimental.mpmd.jax.deepseek.train import (  # noqa: F401  re-exported
    EPS,
    TrainConfig,
    adamw_init,
    adamw_update,
    apply_updates,
    clip_by_global_norm,
    cross_entropy_with_logits,
    global_norm,
    learning_rate_schedule,
    loss_from_logits,
    packed_segment_ids,
    synthetic_batch,
)
from maxtext.experimental.mpmd.jax.qwen35 import model as q_model


def compute_loss(params, batch, config, train_cfg=None, inv_freq=None):
  """`train.py:loss_fn` -- MaxText's masked-mean token cross entropy.

  `batch` is a `synthetic_batch`-shaped dict (targets already next-token
  shifted).  Returns `(loss, metrics)`.
  """
  z_loss = 0.0 if train_cfg is None else train_cfg.z_loss_multiplier
  logits = q_model.forward(
      params, batch["inputs"], batch["inputs_position"], batch["inputs_segmentation"], config, inv_freq=inv_freq
  )
  return loss_from_logits(logits, batch["targets"], batch["targets_segmentation"], config.vocab_size, z_loss)


def train_step(params, opt_state, batch, config, train_cfg, schedule, inv_freq=None):
  """One MaxText training step: loss -> grads -> clip -> AdamW -> apply.

  The update half is `maxtext.experimental.mpmd.jax.deepseek.train.train_step`'s, verbatim.
  """
  grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
  (loss, metrics), raw_grads = grad_fn(params, batch, config, train_cfg, inv_freq)
  grads = clip_by_global_norm(raw_grads, train_cfg.gradient_clipping_threshold)
  updates, opt_state = adamw_update(grads, opt_state, params, train_cfg, schedule)
  params = apply_updates(params, updates)
  metrics = dict(metrics)
  metrics["loss"] = loss
  metrics["raw_grad_norm"] = global_norm(raw_grads)
  metrics["grad_norm"] = global_norm(grads)
  metrics["param_norm"] = global_norm(params)
  return params, opt_state, metrics


__all__ = [
    "EPS",
    "TrainConfig",
    "adamw_init",
    "adamw_update",
    "apply_updates",
    "clip_by_global_norm",
    "compute_loss",
    "cross_entropy_with_logits",
    "global_norm",
    "learning_rate_schedule",
    "loss_from_logits",
    "packed_segment_ids",
    "synthetic_batch",
    "train_step",
]

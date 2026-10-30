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

"""m3's own training stack, built on `nnx.Optimizer`.

Modules:
  * `loss.py` — token-weighted cross-entropy, `jax.checkpoint` vocabulary tiling, z-loss, and MoE auxiliary loss.
  * `step.py` — pure-NNX train/eval steps, global-token microbatch accumulation, and `RouterBias` updates.
  * `loop.py` — minimal synthetic profiling loop and shared `setup_model` composition site.
  * `trainer.py` — production `M3Trainer` and `< 90s` CPU `compile_aot` driver.

m3 models are drivable by two things: this stack, and the existing
`maxtext.training_engine.maxtext_engine` (which the RL stack targets). Anything
that makes a model only usable by one of the two drivers belongs here or in a
bridge, not in a model file.
"""

from maxtext.m3.train.loss import (
    LossBreakdown,
    compute_batch_loss_sums,
    compute_dense_xent_and_z_loss_sums,
    compute_vocab_tiled_xent_sum,
)
from maxtext.m3.train.step import (
    accumulated_loss_and_grads,
    apply_router_bias_update,
    make_eval_step,
    make_train_step,
    reshape_batch_to_microbatches,
)
from maxtext.m3.train.trainer import (
    AOTCompileResult,
    M3Trainer,
    TrainResult,
    build_lr_schedule,
    build_m3_optimizer,
    compile_from_pyconfig,
    train_from_pyconfig,
)

__all__ = [
    "AOTCompileResult",
    "LossBreakdown",
    "M3Trainer",
    "TrainResult",
    "accumulated_loss_and_grads",
    "apply_router_bias_update",
    "build_lr_schedule",
    "build_m3_optimizer",
    "compile_from_pyconfig",
    "compute_batch_loss_sums",
    "compute_dense_xent_and_z_loss_sums",
    "compute_vocab_tiled_xent_sum",
    "make_eval_step",
    "make_train_step",
    "reshape_batch_to_microbatches",
    "train_from_pyconfig",
]

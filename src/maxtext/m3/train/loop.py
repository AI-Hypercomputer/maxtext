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

"""Minimal m3 training loop, intended for collecting profiles.

Deliberately bare: synthetic data, AdamW at a constant learning rate, plain
next-token cross entropy, no checkpointing, no eval, no metrics writer. It
exists so an m3 model can be stepped end to end and traced; it is not a
replacement for `maxtext.trainers.pre_train.train`.

Configuration reuses the legacy `pyconfig` surface, because that is what m3
model constructors take today. The fields read here are `steps`,
`per_device_batch_size`, `max_target_length`, `learning_rate`, the `adam_*`
fields, `gradient_clipping_threshold`, `init_weights_seed`, and the profiler
fields consumed by `maxtext.common.profiler.Profiler` (`profiler`,
`skip_first_n_steps_for_profiler`, `profiler_steps`, ...). The trace is
written under `config.tensorboard_dir`.

Example (CPU or TPU)::

    python3 -m maxtext.m3.train.loop src/maxtext/configs/base.yml \\
      model_name=qwen3-0.6b use_m3_model=true scan_layers=false \\
      per_device_batch_size=1 max_target_length=256 steps=8 \\
      profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=3 \\
      base_output_directory=/tmp/m3 run_name=qwen3_profile \\
      enable_checkpointing=false
"""

import time
from collections.abc import Sequence

from absl import app
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from maxtext.common import profiler
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.core.sharding import create_mesh
from maxtext.utils import max_logging

# Mesh axes the batch dimension is split over. Matches the defaults of
# `create_mesh` (fsdp within a slice, data across slices).
_BATCH_AXES = ("data", "fsdp")


def make_optimizer(config) -> optax.GradientTransformation:
  """AdamW at a constant learning rate, with optional global-norm clipping."""
  adamw = optax.adamw(
      learning_rate=config.learning_rate,
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
  )
  if config.gradient_clipping_threshold > 0:
    return optax.chain(optax.clip_by_global_norm(config.gradient_clipping_threshold), adamw)
  return adamw


def make_synthetic_batch(config, mesh, key: jax.Array) -> dict[str, jax.Array]:
  """Random token batch, sharded along the batch dimension.

  Targets are the inputs shifted left by one, so the loss is ordinary
  next-token prediction rather than a constant.
  """
  batch_size = config.global_batch_size_to_train_on
  seq_len = config.max_target_length
  tokens = jax.random.randint(key, (batch_size, seq_len + 1), 0, config.vocab_size, dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
  batch = {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": positions}
  return jax.device_put(batch, NamedSharding(mesh, P(_BATCH_AXES)))


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  """Mean next-token cross entropy, computed in float32."""
  logits = model(batch["inputs"], batch["positions"])
  losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch["targets"])
  return jnp.mean(losses)


@nnx.jit(donate_argnums=(0, 1))
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]) -> jax.Array:
  """One forward/backward/update. Mutates `model` and `optimizer` in place."""
  loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
  optimizer.update(model, grads)
  return loss


def _log_step(step: int, loss: jax.Array, last_done: float, tokens_per_step: int) -> tuple[float, float]:
  """Blocks until `step` finishes, logs it, and returns its loss and completion time."""
  loss = float(loss)
  done = time.perf_counter()
  step_time = done - last_done
  max_logging.log(f"step {step}: loss {loss:.4f}, {step_time:.3f}s, {tokens_per_step / step_time:,.0f} tokens/s")
  return loss, done


def train_loop(config) -> float:
  """Builds the model and optimizer, runs `config.steps` steps, and returns the last loss."""
  mesh = create_mesh()
  max_logging.log(f"m3 loop: mesh {dict(mesh.shape)} over {len(jax.devices())} device(s)")

  with jax.set_mesh(mesh):
    model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(config.init_weights_seed))
    optimizer = nnx.Optimizer(model, make_optimizer(config), wrt=nnx.Param)

    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    tokens_per_step = config.global_batch_size_to_train_on * config.max_target_length
    max_logging.log(
        f"m3 loop: {config.model_name}, {num_params / 1e9:.3f}B params, "
        f"batch {config.global_batch_size_to_train_on} x {config.max_target_length} tokens"
    )

    prof = profiler.Profiler(config, offset_step=0)
    data_key = jax.random.key(config.init_weights_seed)
    loss = float("nan")
    # Steps are logged one iteration late. Reading a loss blocks until its step
    # finishes, so doing it right after dispatch would leave the device idle
    # while the host prepares the next step. Reading step N-1's loss after
    # dispatching step N keeps one step queued. Step time is the interval
    # between consecutive completions, so step 0's includes compilation.
    pending = None  # (step, loss) dispatched but not yet logged.
    last_done = time.perf_counter()
    for step in range(config.steps):
      batch = make_synthetic_batch(config, mesh, jax.random.fold_in(data_key, step))
      prof.maybe_activate_profiler(step, nnx.state(model))
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        step_loss = train_step(model, optimizer, batch)
      if pending is not None:
        loss, last_done = _log_step(*pending, last_done, tokens_per_step)
      pending = (step, step_loss)
      prof.maybe_deactivate_profiler(step, nnx.state(model))
    if pending is not None:
      loss, last_done = _log_step(*pending, last_done, tokens_per_step)

    # Covers runs that end before the profiling window closes.
    prof.deactivate()
    if config.profiler:
      max_logging.log(f"m3 loop: profile written under {config.tensorboard_dir}")
    return loss


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize(argv)
  train_loop(config)


if __name__ == "__main__":
  app.run(main)

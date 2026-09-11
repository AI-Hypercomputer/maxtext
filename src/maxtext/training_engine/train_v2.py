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

"""A basic pre-training loop driven by `MaxTextTrainingEngine`.

`trainers/pre_train/train.py` is the production pre-training driver; this is a deliberately
small one that exercises the training engine on the same input pipeline. Its purpose is to
run the engine in the shape RL will drive it -- many `fwd_bwd` calls between `update` calls --
against real data rather than the synthetic batches the parity rigs use.

What it does NOT do, by design: elastic retry, goodput recording, batch-size rampup, context
parallelism reordering, evaluation, DiLoCo, or multi-host data expansion. Reach for
`trainers/pre_train/train.py` if any of those matter.

Profiling is deliberately absent here. The engine profiles itself: `fwd_bwd` opens and closes
xprof windows measured in micro-steps (see `training_engine/micro_step_profiler.py`) and emits the step
trace annotations. A `jax.profiler.StepTraceAnnotation` around the loop below would nest a
second "train" step marker inside the engine's own and leave xprof unable to resolve either,
so the loop stays out of the way and the window is configured entirely through
`profiler`/`skip_first_n_steps_for_profiler`/`profiler_steps`.

Example:
  python3 -m maxtext.training_engine.train_v2 src/maxtext/configs/base.yml \
      run_name=engine_smoke model_name=llama3.1-8b dataset_type=synthetic \
      steps=30 per_device_batch_size=1 gradient_accumulation_steps=8 \
      enable_checkpointing=false base_output_directory=gs://my-bucket/out \
      profiler=xplane skip_first_n_steps_for_profiler=40 profiler_steps=8
"""

from collections.abc import Iterator, Sequence
import os
from typing import Any

from absl import app
import jax

import pathwaysutils
import tensorflow as tf

from maxtext.common.data_loader import DataLoader
from maxtext.configs import pyconfig
from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.training_engine import maxtext_engine
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils


def _num_micro_batches(config: pyconfig.HyperParameters) -> int:
  """Returns how many `fwd_bwd` calls make up one optimizer step."""
  return max(config.gradient_accumulation_steps, 1)


def _micro_batch_rows(config: pyconfig.HyperParameters) -> int:
  """Returns the leading dimension of a single micro-batch."""
  return config.micro_batch_size_to_train_on


def _check_supported(config: pyconfig.HyperParameters) -> None:
  """Rejects the configurations this basic driver does not implement.

  Each of these changes the shape or the routing of a loaded batch, so silently ignoring one
  would train on the wrong data rather than fail.
  """
  if config.enable_diloco:
    raise NotImplementedError("train_v2 does not support DiLoCo; the loaded batch gains a replica axis.")
  if config.expansion_factor_real_data > 1:
    raise NotImplementedError(
        "train_v2 does not support expansion_factor_real_data > 1; it loads more rows per host than it trains on."
    )
  if config.gradient_accumulation_steps > 1 and config.global_batch_size_to_load % config.gradient_accumulation_steps:
    raise ValueError(
        f"global_batch_size_to_load={config.global_batch_size_to_load} is not divisible by "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}."
    )


def setup_dataloader(config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh) -> DataLoader:
  """Builds the training dataloader from MaxText's standard input pipeline.

  Args:
    config: MaxText configuration.
    mesh: The SPMD device mesh, used to shard each loaded batch.

  Returns:
    A `DataLoader` whose `load_next_batch()` yields one *global* batch, i.e. all
    `gradient_accumulation_steps` micro-batches concatenated along the leading axis.
  """
  # The eval iterator is discarded: this driver has no evaluation loop.
  data_iterator, _ = create_data_iterator(config, mesh)
  return DataLoader(config, mesh, data_iterator, goodput_recorder=None)


def micro_batches(
    config: pyconfig.HyperParameters,
    batch: Any,
    batch_sharding: jax.sharding.Sharding,
) -> Iterator[Any]:
  """Splits one loaded global batch into the micro-batches `fwd_bwd` consumes.

  The engine accumulates gradients across separate `fwd_bwd` calls rather than scanning over
  micro-batches inside one jitted step, which is what `trainers/pre_train/train.py` does via
  `gradient_accumulation_loss_and_grad`. So the split happens here, in Python, and every
  micro-batch is a separate dispatch -- the same shape RL drives the engine in, and the reason
  the profiler counts micro-steps.

  Takes the batch *before* it reaches the devices and puts each micro-batch there itself. The
  obvious alternative -- shard the global batch once and slice it -- does not work: a slice of
  an array sharded over `fsdp` does not line up with the micro-batch's own sharding, so JAX
  hands back a replicated array and the kernel rejects it ("compiled for input shardings ...
  that disagree"). Correcting that with a `device_put` afterwards would compile a resharding
  collective into every micro-step, which then shows up in every profile.

  Args:
    config: MaxText configuration.
    batch: One host-side global batch, from `DataLoader.load_next_batch_pre_sharding()`.
    batch_sharding: Sharding to place each micro-batch on. Shape-independent, so the one
      built for the global batch is the right one here too.

  Yields:
    `gradient_accumulation_steps` batches of `micro_batch_size_to_train_on` rows each.
  """
  rows = _micro_batch_rows(config)
  for index in range(_num_micro_batches(config)):
    start = index * rows
    # Bind the bounds as defaults: the lambda outlives this iteration of the loop.
    micro_batch = jax.tree.map(lambda leaf, s=start, e=start + rows: leaf[s:e], batch)
    yield jax.device_put(micro_batch, batch_sharding)


def shaped_micro_batch(config: pyconfig.HyperParameters, batch_sharding: jax.sharding.Sharding) -> Any:
  """Returns the avals of one micro-batch, for `engine.compile`.

  `get_shaped_batch` describes the *loaded* batch, so its leading dimension covers every
  micro-batch. Compiling against that would build a kernel for a batch `fwd_bwd` never sees,
  and the first real micro-batch would immediately recompile.
  """
  shaped = maxtext_utils.get_shaped_batch(config, batch_sharding=batch_sharding)
  rows = _micro_batch_rows(config)
  return jax.tree.map(
      lambda aval: jax.ShapeDtypeStruct((rows,) + aval.shape[1:], aval.dtype, sharding=aval.sharding),
      shaped,
  )


def train_loop(config: pyconfig.HyperParameters) -> maxtext_engine.MaxTextTrainingEngine:
  """Runs the pre-training loop to `config.steps` and returns the engine.

  Args:
    config: MaxText configuration.

  Returns:
    The engine, already closed.
  """
  _check_supported(config)

  mesh = maxtext_utils.get_mesh_from_config(config)
  engine = maxtext_engine.MaxTextTrainingEngine(training_config=config, mesh=mesh)

  engine.restore_checkpoint()
  start_step = engine.train_step
  train_utils.validate_completed_steps(start_step, config.steps)

  data_loader = setup_dataloader(config, mesh)

  data_sharding = sharding.get_input_data_sharding(config, mesh)
  engine.compile(shaped_micro_batch(config, data_sharding))

  if start_step == 0:
    max_utils.print_mem_stats("After params initialized")

  max_logging.log(
      f"Training from step {start_step} to {config.steps}, "
      f"{_num_micro_batches(config)} micro-batches per step "
      f"({_micro_batch_rows(config)} rows each)."
  )

  try:
    while engine.train_step < config.steps:
      # `load_next_batch` would shard the whole global batch, which is the one thing
      # `micro_batches` cannot slice. Take it host-side and let that do the placement.
      batch = data_loader.load_next_batch_pre_sharding()
      for micro_batch in micro_batches(config, batch, data_sharding):
        engine.fwd_bwd(micro_batch)
      step = engine.update()

      if step % config.log_period == 0:
        max_logging.log(f"step {step}, {engine.total_micro_steps} micro-steps so far")

      if config.enable_checkpointing and config.checkpoint_period > 0 and step % config.checkpoint_period == 0:
        engine.save_checkpoint(metadata={"step": step, "source": "train_v2"}, step=step)
  finally:
    # `close()` writes the final checkpoint itself when checkpointing is on, so there is no
    # completion save above.
    # It also stops an open profiler window, so a run that ends mid-window still writes its
    # trace, and it drains the metrics the throttler is still holding.
    engine.close()

  return engine


def initialize(argv: Sequence[str]) -> pyconfig.HyperParameters:
  """Parses the config and applies the process-wide JAX/TF settings pre-training needs."""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  tf.config.set_visible_devices([], "GPU")
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_utils.validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  jax.config.update("jax_remove_size_one_mesh_axis_from_type", config.remove_size_one_mesh_axis_from_type)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  return config


def main(argv: Sequence[str]) -> None:
  train_loop(initialize(argv))


if __name__ == "__main__":
  app.run(main)

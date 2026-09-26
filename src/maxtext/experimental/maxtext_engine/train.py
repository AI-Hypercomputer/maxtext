# Copyright 2023-2026 Google LLC
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

"""Training driven by `MaxTextTrainingEngine`.

Training:
  python3 -m maxtext.experimental.maxtext_engine.train src/maxtext/configs/base.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${MODEL_NAME} load_parameters_path=${CHECKPOINT_PATH} \
    tokenizer_path=${TOKENIZER_PATH} \
    per_device_batch_size=1 max_target_length=1024 \
    steps=10 gradient_accumulation_steps=3
"""

from collections.abc import Iterator, Sequence
import itertools
import os
import time
from typing import Any

from absl import app
from flax.linen import logical_axis_rules
import jax
import jax.numpy as jnp
from maxtext.common.data_loader import DataLoader
from maxtext.common.profiler import Profiler
from maxtext.configs import pyconfig
from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.training_engine import maxtext_engine
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import train_utils
import pathwaysutils


def _split_global_batch(
    batch: dict[str, Any],
    num_micro_steps: int,
    data_sharding: jax.sharding.Sharding,
) -> list[dict[str, Any]]:
  """Splits one global batch into `num_micro_steps` contiguous micro-batches."""
  micro_batches: list[dict[str, Any]] = [{} for _ in range(num_micro_steps)]
  for name, value in batch.items():
    shape = jnp.shape(value)
    if not shape:
      for k in range(num_micro_steps):
        micro_batches[k][name] = value
      continue
    micro_size = value.shape[0] // num_micro_steps
    for k in range(num_micro_steps):
      micro_batches[k][name] = jax.device_put(value[k * micro_size : (k + 1) * micro_size], data_sharding)
  return micro_batches


def micro_batch_stream(
    data_loader: DataLoader,
    num_micro_steps: int,
) -> Iterator[dict[str, Any]]:
  """Yields micro-batches forever, one per `fwd_bwd` call."""
  if num_micro_steps == 1:
    while True:
      yield data_loader.load_next_batch()
  else:
    while True:
      yield from _split_global_batch(
          data_loader.load_next_batch(),
          num_micro_steps,
          data_loader.input_data_shardings,
      )


def run_training_loop(
    config: pyconfig.HyperParameters,
    engine: maxtext_engine.MaxTextTrainingEngine,
    mesh: jax.sharding.Mesh,
) -> None:
  """Drives `engine` over `config.steps` optimizer steps."""

  data_iterator, _ = create_data_iterator(config, mesh)
  data_loader = DataLoader(config, mesh, data_iterator, goodput_recorder=None)

  if config.enable_checkpointing:
    engine.restore_checkpoint()

  start_step = engine.train_step
  train_utils.validate_completed_steps(start_step, config.steps)
  if start_step > 0:
    max_logging.log(
        f"WARNING: resuming at step {start_step}, but the data iterator is not"
        " checkpointed and restarts from the beginning of the dataset."
    )
  num_micro_steps = config.gradient_accumulation_steps

  stream = micro_batch_stream(data_loader, num_micro_steps)
  first_batch = next(stream)
  engine.compile(first_batch)
  stream = itertools.chain([first_batch], stream)

  # `MaxTextTrainingEngine._profiler` (`MicroStepProfiler`) only wraps `fwd_bwd` and
  # counts `skip_first_n_steps_for_profiler` / `profiler_steps` in micro-steps rather
  # than optimizer steps. Disable it here and use MaxText's standard `Profiler` around
  # the outer loop so `skip_first_n_steps_for_profiler` and `profiler_steps` capture
  # full optimizer steps (`N * fwd_bwd + 1 * update`) identically to `pre_train/train.py`.
  # pylint: disable=protected-access
  engine._profiler.do_not_profile = True
  prof = Profiler(config, offset_step=start_step)

  # Precompute per-step token count and per-device TFLOPs for throughput logging.
  num_devices = jax.device_count()
  tokens_per_step = config.per_device_batch_size * num_devices * num_micro_steps * config.max_target_length
  total_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)

  max_logging.log(
      f"Starting training engine loop at step {start_step}, running to"
      f" {config.steps} ({num_micro_steps} micro-step(s) per update,"
      f" {tokens_per_step:.0f} tokens/step across {num_devices} devices,"
      f" {total_tflops:.2f} TFLOPs/step/device)."
  )

  try:
    for step in range(start_step, config.steps):
      prof.maybe_activate_profiler(step, engine.state)
      t0 = time.perf_counter()
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        for _ in range(num_micro_steps):
          engine.fwd_bwd(next(stream))
        engine.update()
      # Wait for the step's asynchronous device execution (`update`) and metrics flush
      # to complete before measuring step wall time or closing the profiler window.
      engine._throttler.wait_for_all()
      step_time_s = max(time.perf_counter() - t0, 1e-9)
      tps_per_chip = tokens_per_step / step_time_s / num_devices
      tflops_per_dev = total_tflops / step_time_s
      max_logging.log(
          f"completed step: {step}, seconds: {step_time_s:.3f},"
          f" train_step_time_ms: {step_time_s * 1000:.2f},"
          f" TFLOP/s/device: {tflops_per_dev:.3f},"
          f" tokens/s/chip: {tps_per_chip:.2f}"
      )
      prof.maybe_deactivate_profiler(step, engine.state)
      if config.enable_checkpointing:
        engine.save_checkpoint(metadata=None)
  finally:
    engine.close()


def main(argv: Sequence[str]) -> None:
  """Main function to run training."""
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  mesh = maxtext_utils.get_mesh_from_config(config)
  with logical_axis_rules(config.logical_axis_rules):
    engine = maxtext_engine.MaxTextTrainingEngine(config, mesh=mesh)
    run_training_loop(config, engine, mesh)


if __name__ == "__main__":
  app.run(main)

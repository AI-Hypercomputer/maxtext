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

from collections.abc import Callable, Iterator, Sequence
import functools
import itertools
import os
import time
from typing import Any

from absl import app
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


@functools.cache
def _micro_batch_splitter(
    num_micro_steps: int,
    data_sharding: jax.sharding.Sharding,
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
  """Returns the jitted split behind `_split_global_batch`, cached per `(num_micro_steps, data_sharding)`."""

  def split(batch: dict[str, Any]) -> list[dict[str, Any]]:
    # As `reshape_to_microbatch_accumulations` in `gradient_accumulation.py`:
    # [B, ...] -> [B // G, G, ...] -> [G, B // G, ...], then one micro-batch per index of axis 0.
    stacked = jax.tree.map(
        lambda v: jnp.swapaxes(jnp.reshape(v, (v.shape[0] // num_micro_steps, num_micro_steps) + v.shape[1:]), 0, 1),
        batch,
    )
    return [jax.tree.map(lambda v, k=k: v[k], stacked) for k in range(num_micro_steps)]

  # Not donated: no output has the global batch's shape, so XLA could not reuse its buffers.
  return jax.jit(split, out_shardings=data_sharding)


def _split_global_batch(
    batch: dict[str, Any],
    num_micro_steps: int,
    data_sharding: jax.sharding.Sharding,
) -> list[dict[str, Any]]:
  """Splits one global batch into `num_micro_steps` micro-batches, in `pre_train/train.py`'s order.

  Micro-batch k is rows `k::num_micro_steps`, as `gradient_accumulation.py` builds it. The order
  matters: under `per_device_batch_size < 1`, train.py's `loss_fn` trains on only the first
  `micro_batch_size_to_train_on` rows of each micro-batch.

  Every field is batch-leading and carries `data_sharding`, as `DataLoader.load_next_batch`
  places it. On such an array the strided split keeps each device's rows local, so it compiles
  to no collectives, whereas a contiguous split would move rows between devices.
  """
  return _micro_batch_splitter(num_micro_steps, data_sharding)(batch)


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


def count_chips(devices: Sequence[Any]) -> int:
  """Returns how many physical chips `devices` span.

  On v7x, JAX exposes each of a chip's two TensorCores as its own device with the chip's
  `coords`, so a chip is a distinct `(slice_index, coords)` pair; `coords` repeat across the
  slices of a multislice run. Devices without `coords` (CPU, GPU) are counted as one chip each,
  with a warning.
  """
  chips = set()
  for device in devices:
    coords = getattr(device, "coords", None)
    if coords is None:
      max_logging.log(
          f"WARNING: {device} has no `coords`, so each of the {len(devices)} devices is counted as"
          " one chip in Tokens/s/chip."
      )
      return len(devices)
    chips.add((getattr(device, "slice_index", 0), tuple(coords)))
  return len(chips)


def log_peak_memory() -> None:
  """Logs local device 0's `peak_bytes_in_use` in GiB, if the backend reports it.

  On TPU this counter excludes XLA program temporaries, so it is not the HBM peak.
  """
  device = jax.local_devices()[0]
  peak = (device.memory_stats() or {}).get("peak_bytes_in_use")
  if peak is None:
    return
  max_logging.log(
      f"Peak live arrays on {device}: {peak / 2**30:.2f} GiB"
      " [peak_bytes_in_use; excludes XLA program temporaries on TPU, so not the HBM peak]"
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
  # Live-array memory is logged here and again after the first optimizer step.
  max_utils.print_mem_stats("After engine compile")
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
  # Per-device and per-chip throughput differ where a chip holds more than one JAX device (v7x).
  # `Tokens/s/device` is the label `pre_train/train.py` logs.
  num_chips = count_chips(jax.devices())
  tokens_per_step = config.per_device_batch_size * num_devices * num_micro_steps * config.max_target_length
  total_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)

  max_logging.log(
      f"Starting training engine loop at step {start_step}, running to"
      f" {config.steps} ({num_micro_steps} micro-step(s) per update,"
      f" {tokens_per_step:.0f} tokens/step across {num_devices} devices on {num_chips} chips,"
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
      tokens_per_s = tokens_per_step / step_time_s
      tflops_per_dev = total_tflops / step_time_s
      max_logging.log(
          f"completed step: {step}, seconds: {step_time_s:.3f},"
          f" train_step_time_ms: {step_time_s * 1000:.2f},"
          f" TFLOP/s/device: {tflops_per_dev:.3f},"
          f" Tokens/s/device: {tokens_per_s / num_devices:.3f},"
          f" Tokens/s/chip: {tokens_per_s / num_chips:.3f}"
      )
      if step == start_step:
        max_utils.print_mem_stats("After first optimizer step")
        # Also logged after the loop; logging it here covers a run that fails in a later step.
        log_peak_memory()
      prof.maybe_deactivate_profiler(step, engine.state)
      if config.enable_checkpointing:
        engine.save_checkpoint(metadata=None)
    log_peak_memory()
  finally:
    engine.close()


def main(argv: Sequence[str]) -> None:
  """Main function to run training."""
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  mesh = maxtext_utils.get_mesh_from_config(config)
  # No outer `logical_axis_rules` or mesh context: the engine binds the rules itself wherever it
  # needs them, because Tunix constructs it without them. Leaving them out here keeps this script
  # from hiding a missing binding.
  engine = maxtext_engine.MaxTextTrainingEngine(config, mesh=mesh)
  run_training_loop(config, engine, mesh)


if __name__ == "__main__":
  app.run(main)

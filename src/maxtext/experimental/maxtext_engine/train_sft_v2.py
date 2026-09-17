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

"""Supervised fine-tuning driven by `MaxTextTrainingEngine`.

Training:
  python3 -m maxtext.experimental.maxtext_engine.train_sft_v2 src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME} base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    model_name=${MODEL_NAME} load_parameters_path=${CHECKPOINT_PATH} \
    tokenizer_path=${TOKENIZER_PATH} \
    per_device_batch_size=1 max_target_length=1024 \
    steps=10 gradient_accumulation_steps=3 use_tunix_gradient_accumulation=false
"""

from collections.abc import Iterator, Sequence
import os
import pathwaysutils
import itertools
from typing import Any

from absl import app

import jax
import jax.numpy as jnp

from maxtext.common.data_loader import DataLoader
from maxtext.configs import pyconfig
from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.training_engine import maxtext_engine
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import max_utils
from maxtext.utils import train_utils


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
  """Yields SFT micro-batches forever, one per `fwd_bwd` call."""
  if num_micro_steps == 1:
    while True:
      yield data_loader.load_next_batch()
  else:
    while True:
      yield from _split_global_batch(data_loader.load_next_batch(), num_micro_steps, data_loader.input_data_shardings)


def run_training_loop(
    config: pyconfig.HyperParameters,
    engine: maxtext_engine.MaxTextTrainingEngine,
    mesh: jax.sharding.Mesh,
) -> None:
  """Drives `engine` over `config.steps` optimizer steps of SFT data."""

  data_iterator, _ = create_data_iterator(config, mesh)
  data_loader = DataLoader(config, mesh, data_iterator, goodput_recorder=None)

  engine.restore_checkpoint()

  start_step = engine.train_step
  train_utils.validate_completed_steps(start_step, config.steps)
  if start_step > 0:
    max_logging.log(
        f"WARNING: resuming at step {start_step}, but the data iterator is not checkpointed and "
        "restarts from the beginning of the dataset."
    )
  num_micro_steps = config.gradient_accumulation_steps

  stream = micro_batch_stream(data_loader, num_micro_steps)
  first_batch = next(stream)
  engine.compile(first_batch)
  stream = itertools.chain([first_batch], stream)

  max_logging.log(
      f"Starting SFT engine loop at step {start_step}, running to {config.steps} "
      f"({num_micro_steps} micro-step(s) per update)."
  )

  try:
    for _ in range(start_step, config.steps):
      for _ in range(num_micro_steps):
        engine.fwd_bwd(next(stream))
      engine.update()
      engine.save_checkpoint(metadata=None)
  finally:
    engine.close()


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training."""
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  mesh = maxtext_utils.get_mesh_from_config(config)
  engine = maxtext_engine.MaxTextTrainingEngine(config, mesh=mesh)
  run_training_loop(config, engine, mesh)


if __name__ == "__main__":
  app.run(main)

"""
This module provides functionality to save top-k teacher logits
for distillation purposes in MaxText.

Example command: 
python3 src/maxtext/trainers/post_train/distillation/save_top_k_teacher_logits.py \
src/maxtext/configs/post_train/distillation.yml \
--top_k=128
"""

import os
import pickle
from typing import Sequence
import argparse
import time
import sys
import tensorflow as tf

import jax
import numpy as np
import functools
from itertools import islice

from absl import app
from MaxText import pyconfig
from maxtext.utils import model_creation_utils
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils import maxtext_utils
from maxtext.utils import max_logging

from jax.experimental import multihost_utils
from array_record.python import array_record_module


def get_top_k_logits(logits: jax.Array, k: int):
  """Extracts the top-k values and their vocabulary indices"""
  top_k_values, top_k_indices = jax.lax.top_k(logits, k)
  return top_k_values, top_k_indices


def get_local_cpu_array(arr):
  """Extracts the local data from a sharded JAX array to a host numpy array."""
  return np.concatenate([np.array(s.data) for s in arr.addressable_shards], axis=0)


def generate_and_save_data(config, k_val, optional_keys):
  """Generates top-k logits from the teacher model and saves them to an ArrayRecord file"""
  devices = jax.devices()
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  # Loading teacher model and dataset iterator
  max_logging.log(f"Loading Teacher Model from {config.load_parameters_path}...")
  teacher_model, _ = model_creation_utils.create_nnx_model(config, mesh=mesh)
  train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)

  output_dir = config.base_output_directory
  if config.run_name:
    output_dir = os.path.join(output_dir, config.run_name)

  if jax.process_index() == 0:
    if not tf.io.gfile.exists(output_dir):
      tf.io.gfile.makedirs(output_dir)

  # Sync all hosts to ensure directory exists before writers open files
  multihost_utils.sync_global_devices("create_output_dir")

  # Each host writes to a unique file based on its process index to avoid write conflicts
  filename = f"teacher_top_k_process_{jax.process_index()}.array_record"
  output_path = os.path.join(output_dir, filename)

  max_logging.log(f"Process {jax.process_index()} writing directly to: {output_path}")
  writer = array_record_module.ArrayRecordWriter(output_path, "group_size:1000")

  max_logging.log(f"Starting Top-K generation loop for {config.steps} steps...")
  loop_start = time.time()
  for step, batch in enumerate(islice(train_iter, config.steps)):
    step_start = time.time()
    tokens = batch["inputs"]
    logits = teacher_model(
        decoder_input_tokens=tokens,
        decoder_positions=batch["inputs_position"],
        enable_dropout=False,
    )
    top_k_vals, top_k_idx = get_top_k_logits(logits, k=k_val)

    # Extract only the local data for this host (Distributed Writing)
    local_vals = get_local_cpu_array(top_k_vals)
    local_idx = get_local_cpu_array(top_k_idx)
    local_tokens = get_local_cpu_array(tokens)

    local_optionals = {key: get_local_cpu_array(batch[key]) for key in optional_keys if key in batch}

    record_dict = {
        "tokens": local_tokens,
        "top_k_logits": local_vals,
        "top_k_indices": local_idx,
    }
    for key, local_val in local_optionals.items():
      record_dict[key] = local_val

    writer.write(pickle.dumps(record_dict))

    if step % 50 == 0:
      max_logging.log(f"Successfully processed step {step} in {time.time() - step_start:.4f}s")

  max_logging.log(f"Generation loop finished in {time.time() - loop_start:.2f}s")

  writer.close()
  max_logging.log(f"Finished writing to {output_path}.")


def main(argv: Sequence[str], local_args):
  # Initialize the global configuration
  global_config = pyconfig.initialize(argv)
  teacher_overrides = global_config.teacher_overrides
  teacher_argv = [argv[0], argv[1]]
  teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

  generate_and_save_data(teacher_config, local_args.top_k, local_args.optional_keys)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--top_k",
      type=int,
      required=False,
      default=128,
      help="Top K value for logits.",
  )
  parser.add_argument(
      "--optional_keys",
      type=str,
      nargs="*",
      default=["inputs_position", "inputs_segmentation", "targets_segmentation", "targets"],
      help="Optional keys to save from teacher logits (space-separated).",
  )
  local_arg, remaining_args = parser.parse_known_args()

  main_wrapper = functools.partial(main, local_args=local_arg)
  app.run(main_wrapper, argv=[sys.argv[0]] + remaining_args)

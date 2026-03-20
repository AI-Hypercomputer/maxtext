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

"""
This module provides functionality to save top-k teacher logits
for distillation purposes in MaxText.

Example command: 
python3 src/maxtext/trainers/post_train/distillation/save_top_k_teacher_logits.py \
src/maxtext/configs/post_train/distillation.yml \
--top_k=128 \
--gcs_upload_path=gs://my-bucket/teacher_logits/
"""

import os
import pickle
from typing import Sequence
import argparse
import time
import sys
import tensorflow as tf

import jax
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


def generate_and_save_data(config, local_args):
  """Generates top-k logits from the teacher model and saves them locally, optionally uploading to GCS."""
  k_val = local_args.top_k
  optional_keys = local_args.optional_keys
  gcs_upload_path = local_args.gcs_upload_path
  local_tmp_dir = local_args.local_tmp_dir

  devices = jax.devices()
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  # Loading teacher model and dataset iterator
  max_logging.log(f"Loading Teacher Model from {config.load_parameters_path}...")
  teacher_model, _ = model_creation_utils.create_nnx_model(config, mesh=mesh)
  train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)

  # Setup local tmp directory for Host 0
  filename = "teacher_top_k_global.array_record"
  local_output_path = os.path.join(local_tmp_dir, filename)

  writer = None
  if jax.process_index() == 0:
    if not os.path.exists(local_tmp_dir):
      os.makedirs(local_tmp_dir)
    max_logging.log(f"Process 0 writing globally gathered data to local path: {local_output_path}")
    writer = array_record_module.ArrayRecordWriter(local_output_path, "group_size:1000")

  # Sync all hosts before starting the loop
  multihost_utils.sync_global_devices("start_generation_loop")

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

    # Fetch the global distributed jax arrays
    global_vals = jax.device_get(top_k_vals)
    global_idx = jax.device_get(top_k_idx)
    global_tokens = jax.device_get(tokens)

    if jax.process_index() == 0:
      record_dict = {
          "tokens": global_tokens,
          "top_k_logits": global_vals,
          "top_k_indices": global_idx,
      }

      for key in optional_keys:
        if key in batch:
          record_dict[key] = jax.device_get(batch[key])

      writer.write(pickle.dumps(record_dict))

      if step % 50 == 0:
        max_logging.log(f"Successfully processed step {step} in {time.time() - step_start:.4f}s")

  max_logging.log(f"Generation loop finished in {time.time() - loop_start:.2f}s")

  # Sync to ensure all hosts finish the forward passes before host 0 starts uploading
  multihost_utils.sync_global_devices("loop_finished")

  # Finalize writing and handle GCS upload on Host 0
  if jax.process_index() == 0:
    writer.close()
    max_logging.log(f"Finished writing to local disk: {local_output_path}")

    if gcs_upload_path:
      gcs_file_path = os.path.join(gcs_upload_path, filename)
      max_logging.log(f"Flag --gcs_upload_path detected. Uploading to: {gcs_file_path}")

      if not tf.io.gfile.exists(gcs_upload_path):
        tf.io.gfile.makedirs(gcs_upload_path)

      # Perform the bulk copy to GCS
      tf.io.gfile.copy(local_output_path, gcs_file_path, overwrite=True)
      max_logging.log("GCS Upload complete.")

  # Sync all hosts one last time so worker hosts don't terminate the job
  multihost_utils.sync_global_devices("upload_complete")


def main(argv: Sequence[str], local_args):
  # Initialize the global configuration
  global_config = pyconfig.initialize(argv)
  teacher_overrides = global_config.teacher_overrides
  teacher_argv = [argv[0], argv[1]]
  teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

  # Pass the entire local_args object to clean up the function signature
  generate_and_save_data(teacher_config, local_args)


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
  parser.add_argument(
      "--gcs_upload_path",
      type=str,
      required=False,
      default=None,
      help="Optional GCS directory (e.g., gs://my-bucket/logits/) to upload the locally saved ArrayRecord file.",
  )
  parser.add_argument(
      "--local_tmp_dir",
      type=str,
      required=False,
      default="/tmp",
      help="Local temporary directory to write the ArrayRecord file before optional GCS upload.",
  )
  local_arg, remaining_args = parser.parse_known_args()

  main_wrapper = functools.partial(main, local_args=local_arg)
  app.run(main_wrapper, argv=[sys.argv[0]] + remaining_args)

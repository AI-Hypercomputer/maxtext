# # Copyright 2023–2026 Google LLC
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """
# This module provides functionality to save top-k teacher logits
# for distillation purposes in MaxText.

# Example command: 
# python3 src/maxtext/trainers/post_train/distillation/save_top_k_teacher_logits.py \
#   src/maxtext/configs/post_train/distillation.yml \
#   --local_tmp_dir=/tmp/save_logits_dir \
#   --steps_per_file=10
# """

# import os
# import subprocess
# from concurrent.futures import ThreadPoolExecutor

# # Force the pure Python protobuf implementation to avoid UPB compatibility issues with TFDS
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# import pickle
# from typing import Sequence
# import argparse
# import time
# import sys
# import tensorflow as tf
# import re
# import numpy as np

# import jax
# import jax.numpy as jnp
# from flax import nnx
# import functools
# from itertools import islice

# from absl import app
# from maxtext.configs import pyconfig
# from maxtext.utils import model_creation_utils
# from maxtext.input_pipeline import input_pipeline_interface
# from maxtext.utils import maxtext_utils
# from maxtext.utils import max_logging

# from jax.experimental import multihost_utils
# from array_record.python import array_record_module


# def get_top_k_logits(logits: jax.Array, k: int):
#   """Extracts the top-k values and their vocabulary indices"""
#   top_k_values, top_k_indices = jax.lax.top_k(logits, k)
#   return top_k_values, top_k_indices


# def get_start_step(config, local_args):
#   """Determines the starting step for the generation process."""
#   if jax.process_index() != 0:
#     return 0

#   output_dir = local_args.gcs_upload_path if local_args.gcs_upload_path else local_args.local_tmp_dir
#   if not tf.io.gfile.exists(output_dir):
#     tf.io.gfile.makedirs(output_dir)
#     return 0

#   existing_files = tf.io.gfile.glob(os.path.join(output_dir, "teacher_top_k_part_*.array_record"))
#   if not existing_files:
#     return 0

#   # Find the highest part number from the filenames
#   max_part_num = max(
#       (int(m.group(1)) for f in existing_files if (m := re.search(r"part_(\d+).array_record", os.path.basename(f)))),
#       default=-1,
#   )

#   if max_part_num == -1:
#     return 0

#   start_step = max_part_num * local_args.steps_per_file
#   max_logging.log(f"Found existing data, resuming from step {start_step}")
#   return start_step
  
# def create_tf_example(example_dict):
#     """Converts a dictionary of single-example numpy arrays to a tf.train.Example."""
#     features = {}
#     for key, val in example_dict.items():
#         # Flatten the arrays since tf.train.Example requires flat lists
#         flat_val = np.asarray(val).flatten()
        
#         if flat_val.dtype in [np.float32, np.float64, np.float16]:
#             features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=flat_val))
#         elif flat_val.dtype in [np.int32, np.int64]:
#             features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=flat_val))
#         else:
#             raise ValueError(f"Unsupported dtype {flat_val.dtype} for key {key}")
            
#     return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
  
# def background_upload(local_path, gcs_path):
#   """Executes a highly optimized concurrent upload via gcloud."""
#   try:
#       subprocess.run(
#           ["gcloud", "storage", "cp", local_path, gcs_path], 
#           check=True, 
#           capture_output=True
#       )
#       # We only delete the file AFTER the upload is successfully finished
#       os.remove(local_path)
#       max_logging.log(f"Background upload complete: {gcs_path}")
#   except subprocess.CalledProcessError as e:
#       max_logging.log(f"Upload failed for {local_path}: {e.stderr.decode()}")
      
# @nnx.jit(static_argnames=("k",))
# def teacher_step(model, batch, k):
#     logits = model(
#         decoder_input_tokens=batch["inputs"],
#         decoder_positions=batch["inputs_position"],
#         decoder_segment_ids=batch.get("inputs_segmentation"),
#         decoder_target_tokens=batch.get("targets"),
#         decoder_target_mask=batch.get("targets_segmentation"),
#         enable_dropout=False,
#     )
#     return get_top_k_logits(logits, k=k)

# def generate_and_save_data(config, local_args):
#   """Generates top-k logits from the teacher model and saves them locally, optionally uploading to GCS"""
#   k_val = local_args.top_k
#   optional_keys = local_args.optional_keys
#   gcs_upload_path = local_args.gcs_upload_path
#   local_tmp_dir = local_args.local_tmp_dir
#   steps_per_file = local_args.steps_per_file
  
#   writer = None
#   local_output_path = None
#   upload_executor = None

#   if jax.process_index() == 0:
#     if not os.path.exists(local_tmp_dir):
#       os.makedirs(local_tmp_dir)
#     # Initialize the thread pool exactly once
#     upload_executor = ThreadPoolExecutor(max_workers=4)

#   devices = jax.devices()
#   devices_array = maxtext_utils.create_device_mesh(config, devices)
#   mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

#   # Loading teacher model and dataset iterator
#   max_logging.log(f"Loading Teacher Model from {config.load_parameters_path}...")
#   teacher_model = model_creation_utils.from_pretrained(config, mesh=mesh)
#   train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)

#   # Determine start_step for resuming
#   start_step = get_start_step(config, local_args)
#   start_step = int(multihost_utils.broadcast_one_to_all(jnp.array(start_step)))

#   # Sync all hosts before starting the loop
#   multihost_utils.sync_global_devices("start_generation_loop")

#   with mesh:
#     max_logging.log(f"Starting Top-K generation loop for {config.steps - start_step} steps...")
#     loop_start = time.time()

#     for step, batch in enumerate(islice(train_iter, start_step, config.steps), start=start_step):
#       step_start = time.time()

#       # Open a new writer for each file chunk on process 0
#       if jax.process_index() == 0 and step % steps_per_file == 0:
#         if writer:
#           writer.close()
#           if gcs_upload_path:
#             gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
#             max_logging.log(f"Queueing background upload for {local_output_path} to {gcs_file_path}")
#             # Submit to background thread (Do NOT call os.remove() here)
#             upload_executor.submit(background_upload, local_output_path, gcs_file_path)

#         file_index = step // steps_per_file
#         filename = f"teacher_top_k_part_{file_index:05d}.array_record"
#         local_output_path = os.path.join(local_tmp_dir, filename)
#         max_logging.log(f"Process 0 writing to new chunk: {local_output_path}")
#         writer = array_record_module.ArrayRecordWriter(local_output_path, "group_size:1000")

#       tokens = batch["inputs"]
      
#       if step == start_step:
#         max_logging.log(f"Confirmed input tokens shape from pipeline: {tokens.shape}")
        
#       top_k_vals, top_k_idx = teacher_step(teacher_model, batch, k_val)

#       # NOTE: process_allgather is a collective operation and MUST be called by all processes
#       # to avoid deadlocks. Therefore, we gather on all devices, but only save on host 0.
#       global_vals = jax.experimental.multihost_utils.process_allgather(top_k_vals, tiled=True)
#       global_idx = jax.experimental.multihost_utils.process_allgather(top_k_idx, tiled=True)
#       global_tokens = jax.experimental.multihost_utils.process_allgather(tokens, tiled=True)

#       optional_data = {}
#       for key in optional_keys:
#         if key in batch:
#           optional_data[key] = jax.experimental.multihost_utils.process_allgather(batch[key], tiled=True)

#       if jax.process_index() == 0 and writer:
#         # Convert jax arrays to numpy for faster slicing and iteration
#         global_tokens_np = np.array(global_tokens)
#         global_vals_np = np.array(global_vals)
#         global_idx_np = np.array(global_idx)
#         optional_data_np = {k: np.array(v) for k, v in optional_data.items()}
        
#         batch_size = global_tokens_np.shape[0]
        
#         # Unbatch: iterate through the batch dimension and write individual examples
#         for i in range(batch_size):
#             example_dict = {
#                 "inputs": global_tokens_np[i],
#                 "top_k_logits": global_vals_np[i],
#                 "top_k_indices": global_idx_np[i],
#             }
#             for key, val_np in optional_data_np.items():
#                 example_dict[key] = val_np[i]
                
#             # Serialize and write
#             serialized_example = create_tf_example(example_dict)
#             writer.write(serialized_example)

#         if step % 50 == 0:
#           max_logging.log(f"Successfully processed step {step} in {time.time() - step_start:.4f}s")
          
#       # Synchronize all hosts after potential write on host 0
#       multihost_utils.sync_global_devices(f"step_{step}_complete")

#     max_logging.log(f"Generation loop finished in {time.time() - loop_start:.2f}s")

#   # Sync to ensure all hosts finish the forward passes before host 0 starts uploading
#   multihost_utils.sync_global_devices("loop_finished")

#   # Finalize writing and handle GCS upload on Host 0
#   if jax.process_index() == 0 and writer:
#     writer.close()
#     max_logging.log(f"Finished writing to local disk: {local_output_path}")

#     if gcs_upload_path:
#       gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
#       max_logging.log(f"Queueing final chunk to: {gcs_file_path}")
#       # Use the thread pool for the final file too!
#       upload_executor.submit(background_upload, local_output_path, gcs_file_path)

#     # Safely close out the thread pool and wait for all uploads to finish before exiting
#     if upload_executor:
#       max_logging.log("Waiting for all background uploads to finish before exiting...")
#       upload_executor.shutdown(wait=True)
#       max_logging.log("All GCS uploads complete.")

#   # Sync all hosts one last time so worker hosts don't terminate the job
#   multihost_utils.sync_global_devices("upload_complete")


# def main(argv: Sequence[str], local_args):
#   # Initialize the global configuration
#   global_config = pyconfig.initialize(argv)
#   teacher_overrides = global_config.teacher_overrides
#   teacher_argv = [argv[0], argv[1]]
#   teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

#   # Pass the entire local_args object to clean up the function signature
#   generate_and_save_data(teacher_config, local_args)


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       "--top_k",
#       type=int,
#       required=False,
#       default=128,
#       help="Top K value for logits.",
#   )
#   parser.add_argument(
#       "--optional_keys",
#       type=str,
#       nargs="*",
#       default=["inputs_position", "inputs_segmentation", "targets_segmentation", "targets"],
#       help="Optional keys to save from teacher logits (space-separated).",
#   )
#   parser.add_argument(
#       "--gcs_upload_path",
#       type=str,
#       required=False,
#       default=None,
#       help="Optional GCS directory (e.g., gs://my-bucket/logits/) to upload the locally saved ArrayRecord file.",
#   )
#   parser.add_argument(
#       "--local_tmp_dir",
#       type=str,
#       required=False,
#       default="/tmp",
#       help="Local temporary directory to write the ArrayRecord file before optional GCS upload.",
#   )
#   parser.add_argument(
#       "--steps_per_file",
#       type=int,
#       default=1000,
#       help="Number of steps to save in each chunk.",
#   )
#   local_arg, remaining_args = parser.parse_known_args()

#   main_wrapper = functools.partial(main, local_args=local_arg)
#   app.run(main_wrapper, argv=[sys.argv[0]] + remaining_args)

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

import os
<<<<<<< HEAD
<<<<<<< HEAD
import safetensors.numpy
=======
=======
import subprocess
from concurrent.futures import ThreadPoolExecutor
>>>>>>> 6b5d9598c (Speed up offline distillation and saving top-k teacher logits)

# Force the pure Python protobuf implementation to avoid UPB compatibility issues with TFDS
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pickle
>>>>>>> a0461ff4f (Updated script to work in multihost setting for collecting and using logits offline)
from typing import Sequence
import argparse
import time
import sys
import tensorflow as tf
import re
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import functools
from itertools import islice

from absl import app
from maxtext.configs import pyconfig
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

def get_start_step(config, local_args):
  """Determines the starting step for the generation process."""
  if jax.process_index() != 0:
    return 0

  output_dir = local_args.gcs_upload_path if local_args.gcs_upload_path else local_args.local_tmp_dir
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
    return 0

  existing_files = tf.io.gfile.glob(os.path.join(output_dir, "teacher_top_k_part_*.array_record"))
  if not existing_files:
    return 0

  # Updated regex to handle the host ID in the filename
  max_part_num = max(
      (int(m.group(1)) for f in existing_files if (m := re.search(r"part_(\d+)_host", os.path.basename(f)))),
      default=-1,
  )

  if max_part_num == -1:
    return 0

  start_step = max_part_num * local_args.steps_per_file
  max_logging.log(f"Found existing data, resuming from step {start_step}")
  return start_step
  
def create_tf_example(example_dict):
    """Converts a dictionary of single-example numpy arrays to a tf.train.Example."""
    features = {}
    for key, val in example_dict.items():
        if key == "sequence_hash":
            features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
            continue
            
        flat_val = np.asarray(val).flatten()
        
        if flat_val.dtype in [np.float32, np.float64, np.float16, jnp.bfloat16]:
            features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=flat_val.astype(np.float32)))
        elif flat_val.dtype in [np.int32, np.int64]:
            features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=flat_val.astype(np.int64)))
        else:
            raise ValueError(f"Unsupported dtype {flat_val.dtype} for key {key}")
            
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

def background_process_and_write(writer, tokens, vals, idx, opt_data):
    """Executes entirely on a background CPU thread so the TPU never waits."""
    tokens_np = np.array(tokens)
    vals_np = np.array(vals)
    idx_np = np.array(idx)
    opt_data_np = {k: np.array(v) for k, v in opt_data.items()}
    
    batch_size = tokens_np.shape[0]
    for i in range(batch_size):
        seq_bytes = tokens_np[i].tobytes()
        example_dict = {
            "inputs": tokens_np[i],
            "top_k_logits": vals_np[i],
            "top_k_indices": idx_np[i],
            "sequence_hash": hash(seq_bytes) 
        }
        for key, val_np in opt_data_np.items():
            example_dict[key] = val_np[i]
            
        writer.write(create_tf_example(example_dict))

def background_upload(local_path, gcs_path, process_index):
  """Executes a highly optimized concurrent upload via gcloud."""
  try:
      subprocess.run(
          ["gcloud", "storage", "cp", local_path, gcs_path], 
          check=True, 
          capture_output=True
      )
      os.remove(local_path)
      if process_index == 0:
          max_logging.log(f"Background upload complete: {gcs_path}")
  except subprocess.CalledProcessError as e:
      if process_index == 0:
          max_logging.log(f"Upload failed for {local_path}: {e.stderr.decode()}")
      
@nnx.jit(static_argnames=("k",))
def teacher_step(model, batch, k):
    logits = model(
        decoder_input_tokens=batch["inputs"],
        decoder_positions=batch["inputs_position"],
        decoder_segment_ids=batch.get("inputs_segmentation"),
        decoder_target_tokens=batch.get("targets"),
        decoder_target_mask=batch.get("targets_segmentation"),
        enable_dropout=False,
    )
    return get_top_k_logits(logits, k=k)

def generate_and_save_data(config, local_args):
  k_val = local_args.top_k
  optional_keys = local_args.optional_keys
  gcs_upload_path = local_args.gcs_upload_path
  local_tmp_dir = local_args.local_tmp_dir
  steps_per_file = local_args.steps_per_file
  
  writer = None
  local_output_path = None
  
  # ALL HOSTS initialize their own directories and thread pools
  if not os.path.exists(local_tmp_dir):
    os.makedirs(local_tmp_dir, exist_ok=True)
  
  upload_executor = ThreadPoolExecutor(max_workers=4)
  write_executor = ThreadPoolExecutor(max_workers=2) 

  devices = jax.devices()
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  if jax.process_index() == 0:
      max_logging.log(f"Loading Teacher Model from {config.load_parameters_path}...")
      
  teacher_model = model_creation_utils.from_pretrained(config, mesh=mesh)
  train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)

  start_step = get_start_step(config, local_args)
  start_step = int(multihost_utils.broadcast_one_to_all(jnp.array(start_step)))

  multihost_utils.sync_global_devices("start_generation_loop")

  with mesh:
    if jax.process_index() == 0:
        max_logging.log(f"Starting Distributed Top-K generation loop for {config.steps - start_step} steps...")
        
    loop_start = time.time()

    for step, batch in enumerate(islice(train_iter, start_step, config.steps), start=start_step):
      step_start = time.time()

      # ALL HOSTS execute the file opening/closing logic
      if step % steps_per_file == 0:
        if writer:
            write_executor.shutdown(wait=True) 
            writer.close()
            if gcs_upload_path:
                gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
                if jax.process_index() == 0:
                    max_logging.log(f"Queueing distributed background uploads for Step {step}...")
                upload_executor.submit(background_upload, local_output_path, gcs_file_path, jax.process_index())
            write_executor = ThreadPoolExecutor(max_workers=2)

        file_index = step // steps_per_file
        # CRITICAL: Filename now includes host ID to prevent GCS collisions
        filename = f"teacher_top_k_part_{file_index:05d}_host_{jax.process_index():03d}.array_record"
        local_output_path = os.path.join(local_tmp_dir, filename)
        writer = array_record_module.ArrayRecordWriter(local_output_path, "group_size:1")

      # tokens = batch["inputs"]
      # top_k_vals, top_k_idx = teacher_step(teacher_model, batch, k_val)

      # # NO ALLGATHER. Everyone keeps their own local TPU slice.
      # optional_data = {}
      # for key in optional_keys:
      #   if key in batch:
      #     optional_data[key] = batch[key]

      # if writer:
      #   # Offload the Protobuf formatting to the local CPU thread
      #   # write_executor.submit(
      #   #     background_process_and_write, 
      #   #     writer, tokens, top_k_vals, top_k_idx, optional_data
      #   # )
      #   background_process_and_write(writer, tokens, top_k_vals, top_k_idx, optional_data)
      
      tokens = batch["inputs"]
      top_k_vals, top_k_idx = teacher_step(teacher_model, batch, k_val)

      # 1. Gather all data to all hosts to make it fully addressable in JAX
      global_tokens = jax.experimental.multihost_utils.process_allgather(tokens, tiled=True)
      global_vals = jax.experimental.multihost_utils.process_allgather(top_k_vals, tiled=True)
      global_idx = jax.experimental.multihost_utils.process_allgather(top_k_idx, tiled=True)

      optional_data = {}
      for key in optional_keys:
        if key in batch:
          optional_data[key] = jax.experimental.multihost_utils.process_allgather(batch[key], tiled=True)

<<<<<<< HEAD
<<<<<<< HEAD
      writer.write(safetensors.numpy.save(record_dict))
=======
      if jax.process_index() == 0:
        record_dict = {
            "tokens": global_tokens,
            "top_k_logits": global_vals,
            "top_k_indices": global_idx,
        }
        for key, val in optional_data.items():
          record_dict[key] = val
>>>>>>> a0461ff4f (Updated script to work in multihost setting for collecting and using logits offline)
=======
      if writer:
        # 2. Convert to numpy safely on the CPU
        global_tokens_np = np.array(global_tokens)
        global_vals_np = np.array(global_vals)
        global_idx_np = np.array(global_idx)
        optional_data_np = {k: np.array(v) for k, v in optional_data.items()}
>>>>>>> 6b5d9598c (Speed up offline distillation and saving top-k teacher logits)

        # 3. Slice out ONLY this host's local fraction of the batch
        global_batch_size = global_tokens_np.shape[0]
        local_batch_size = global_batch_size // jax.process_count()
        start_idx = jax.process_index() * local_batch_size
        end_idx = start_idx + local_batch_size

        local_tokens_np = global_tokens_np[start_idx:end_idx]
        local_vals_np = global_vals_np[start_idx:end_idx]
        local_idx_np = global_idx_np[start_idx:end_idx]
        local_opt_data_np = {k: v[start_idx:end_idx] for k, v in optional_data_np.items()}

        # 4. Write synchronously (takes <0.1 seconds because it's only 1/32 of the data!)
        background_process_and_write(writer, local_tokens_np, local_vals_np, local_idx_np, local_opt_data_np)

      if step % 50 == 0 and jax.process_index() == 0:
          max_logging.log(f"Successfully processed step {step} in {time.time() - step_start:.4f}s")
          
      # Sync hosts briefly to ensure TPU compute stays aligned across the mesh
      multihost_utils.sync_global_devices(f"step_{step}_complete")

    if jax.process_index() == 0:
        max_logging.log(f"Generation loop finished in {time.time() - loop_start:.2f}s")

  multihost_utils.sync_global_devices("loop_finished")

  # Finalize writing and handle GCS upload on ALL Hosts
  if writer:
    if write_executor:
        write_executor.shutdown(wait=True)
    writer.close()

    if gcs_upload_path:
      gcs_file_path = os.path.join(gcs_upload_path, os.path.basename(local_output_path))
      upload_executor.submit(background_upload, local_output_path, gcs_file_path, jax.process_index())

    if upload_executor:
      if jax.process_index() == 0:
          max_logging.log("Waiting for all background uploads to finish across all hosts...")
      upload_executor.shutdown(wait=True)
      if jax.process_index() == 0:
          max_logging.log("All GCS uploads complete.")

  multihost_utils.sync_global_devices("upload_complete")


def main(argv: Sequence[str], local_args):
  global_config = pyconfig.initialize(argv)
  teacher_overrides = global_config.teacher_overrides
  teacher_argv = [argv[0], argv[1]]
  teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)
  generate_and_save_data(teacher_config, local_args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--top_k", type=int, default=128)
  parser.add_argument("--optional_keys", type=str, nargs="*", default=["inputs_position", "inputs_segmentation", "targets_segmentation", "targets"])
  parser.add_argument("--gcs_upload_path", type=str, default=None)
  parser.add_argument("--local_tmp_dir", type=str, default="/tmp")
  parser.add_argument("--steps_per_file", type=int, default=1000)
  local_arg, remaining_args = parser.parse_known_args()

  main_wrapper = functools.partial(main, local_args=local_arg)
  app.run(main_wrapper, argv=[sys.argv[0]] + remaining_args)
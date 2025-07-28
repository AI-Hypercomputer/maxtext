"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Standalone checkpointer - only saves and restores checkpoints at regular intervals, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
from typing import Sequence
import requests
import time
import orbax.checkpoint as ocp

from absl import app

import numpy as np

import jax
from jax import numpy as jnp

from flax.linen import partitioning as nn_partitioning

import MaxText as mt
from MaxText import checkpointing
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import max_logging
from MaxText import pyconfig
from MaxText.train import get_first_step, validate_train_config
from MaxText.layers import models

from google.cloud import storage
import csv
import uuid

Composite = ocp.args.Composite
Transformer = models.Transformer

CHECKPOINT_RESTORE_TIME_DIRECTORY = "ckpt_restore_time"
CHECKPOINT_WRITE_TIME_DIRECTORY = "ckpt_write_time"
NODE_ATTRIBUTES_TIME_DIRECTORY = "node-attributes"


def upload_csv(bucket_name, object_name, data):
  tmp_file_name = f"{str(uuid.uuid4())}.csv"
  with open(tmp_file_name, 'w', encoding="utf-8", newline='') as file:
    csv_writer = csv.writer(file)
    for _, t in enumerate(data):
      csv_writer.writerow(t)
  
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(object_name)
  # Set if_generation_match to 0 for the object insertion to be idempotent
  # such that server-side errors will be auto retried. See
  # https://cloud.google.com/storage/docs/retry-strategy#idempotency-operations and
  # https://cloud.google.com/storage/docs/retry-strategy#tools.
  generation_match_precondition = 0
  try:
    blob.upload_from_filename(tmp_file_name, if_generation_match=generation_match_precondition)
  except Exception as e:
    max_logging.log(f"An error occurred during upload {object_name} to {bucket_name}: {e}")
  

def checkpoint_loop(config, state=None):
  """Main Checkpointing loop.
  Saves checkpoints.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  if config.gcs_metrics_bucket:
    max_logging.log(f"Uploading node attributes to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    base_name = f"{jax.process_index()}.csv"
    node_attributes = {
        'rank': jax.process_index(),
        'pod_name': os.environ.get('MY_POD_NAME', ''),
        'pod_ip': os.environ.get('MY_POD_IP', ''),
        'node_name': os.environ.get('MY_NODE_NAME', ''),
        'node_ip': os.environ.get('MY_NODE_IP', ''),
        'compute_instance_id': get_compute_instance_id(),
    }
    
    upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, NODE_ATTRIBUTES_TIME_DIRECTORY, base_name), 
        [list(node_attributes.keys()), list(node_attributes.values())] # convert into headers and values
    )
  
  max_logging.log(f"Finished uploading node attributes to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()} for run {config.run_name}")
  ckpt_read_time = []
  ckpt_read_time.append(["rank", "checkpoint_num", "checkpoint_restore_time"]) # header for checkpoint restore metrics.
  ckpt_write_time = []
  ckpt_write_time.append(["rank", "checkpoint_num", "checkpoint_write_time"]) # header for checkpoint write metrics.

  model = mt.from_pretrained(config)
  mesh = model.mesh
  init_rng, checkpoint_manager, _, tx = train_utils.create_training_tools(config, model, mesh)

  unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")
  checkpoint_load_start = datetime.datetime.now()
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, _ = checkpointing.load_state_if_possible(
        checkpoint_manager,
        None,
        config.load_parameters_path,
        config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb,
        unboxed_abstract_state,
        enable_single_replica_ckpt_restoring=config.enable_single_replica_ckpt_restoring,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    if state:
      state = state["items"]

  jax.block_until_ready(state)
  checkpoint_load_end = datetime.datetime.now()
  if state is not None:  # Checkpoint was available for restore
    if jax.process_index() == 0:
      max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint restored in : {checkpoint_load_end - checkpoint_load_start}")
  else:  # Checkpoint was unavailable, state needs to be initialized
    state, _, _, _ = maxtext_utils.setup_training_state(model, None, tx, config, init_rng, mesh, checkpoint_manager)
  state = add_entropy_to_checkpoint(state)

  ckpt_read_idx = 0
  ckpt_write_idx = 0
  start_step = get_first_step(state)  # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    start_time = datetime.datetime.now()
    if checkpoint_manager is not None:
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      start_time = datetime.datetime.now()
      if checkpointing.save_checkpoint(checkpoint_manager, int(step), state):
        checkpoint_manager.wait_until_finished()
        end_time = datetime.datetime.now()
        checkpoint_write_time = (end_time - start_time).total_seconds()
        ckpt_write_time.append([jax.process_index(), ckpt_write_idx, checkpoint_write_time])
        ckpt_write_idx += 1
        if jax.process_index() == 0:
          max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint saved in {end_time - start_time}, step {step}, on host 0")
    if jax.process_index() == 0:
      elapsed_time = datetime.datetime.now() - start_time
      time_to_wait = config.per_step_interval - elapsed_time.total_seconds()
      if time_to_wait > 0:
        max_logging.log(f"Waiting {time_to_wait} seconds to reach step time of {config.per_step_interval} seconds for step {step}")
        time.sleep(time_to_wait)
    jax.experimental.multihost_utils.sync_global_devices("Barrier after step")
    
    # Restore from the checkpoint that was just saved.
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      try:
        state = checkpoint_manager.restore(
              step,
              args=ocp.args.Composite(items=ocp.args.PyTreeRestore(item=unboxed_abstract_state)),
            )
        if state:
          state = state["items"]
      except FileNotFoundError:
        # No checkpoint was found for the step, presumably because one was not produced for the step. Continue on.
        continue
      jax.block_until_ready(state)
      end_time = datetime.datetime.now()
      checkpoint_restore_time = (end_time - start_time).total_seconds()
      ckpt_read_time.append([jax.process_index(), ckpt_read_idx, checkpoint_restore_time])
      ckpt_read_idx += 1
      if jax.process_index() == 0:
        max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint restored in {end_time - start_time} ,step {step}, on host 0")
  
  if config.gcs_metrics_bucket:
    base_name = f"{jax.process_index()}.csv"
    max_logging.log(f"Uploading write metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, CHECKPOINT_WRITE_TIME_DIRECTORY, base_name), 
        ckpt_write_time
    )
    max_logging.log(f"Finished uploading write metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}  for run {config.run_name}")
    
    max_logging.log(f"Uploading restore metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, CHECKPOINT_RESTORE_TIME_DIRECTORY, base_name), 
        ckpt_read_time
    )
    max_logging.log(f"Finished uploading restore metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()} for run {config.run_name}")

  # Make sure that all the additional checkpoints are deleted before exiting.
  checkpoint_manager.close()
  # The need of a super long barrer operation:
  #  1. Only the rank 0 is doing the background delete and all other ranks will exit the workload.
  #  2. Other ranks exiting the workload causes the JAX distributed runtime to wait for rank 0 to exit.
  #  3. The wait is currently at 5 min and not easily modified (requires a rebuild of the JAX dependency).
  # Therefore, we use a barrier with a crazy long timeout to make sure that rank 0 can delete all the
  # checkpoints and signal the other processes to exit.
  client = jax._src.distributed.global_state.client
  client.wait_at_barrier(barrier_id="waiting for deletion to complete", timeout_in_ms=config.final_ckpts_deletion_timeout_in_s * 1000)

  return state

def add_entropy_to_checkpoint(state):
  """Introduce randomness in checkpoints. This is useful to simulate real checkpoints, without training.
  Args:
    state: Initial state
  Returns:
    state: Returns state with entropy added to the optimizer state.
  """
  opt_0 = state.opt_state[0]
  opt_0 = opt_0._replace(mu=jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), state.params))
  opt_0 = opt_0._replace(nu=jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), state.params))
  new_opt = [opt_0] + list(state.opt_state[1:])
  state = state.replace(opt_state=new_opt)
  return state

def get_compute_instance_id():
  """Get the compute instance id from the metadata server."""
  metadata_server_url = (
      'http://metadata.google.internal/computeMetadata/v1/instance/id'
  )
  headers = {'Metadata-Flavor': 'Google'}

  try:
    response = requests.get(metadata_server_url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes

    instance_id = response.text
    return instance_id

  except requests.exceptions.RequestException as e:
    max_logging.log('Error fetching instance ID:', e)
  return ''

def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_cpu_enable_gloo_collectives", True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  validate_train_config(config)
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  checkpoint_loop(config)


if __name__ == "__main__":
  app.run(main)

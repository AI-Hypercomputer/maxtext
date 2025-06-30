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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import time
from MaxText.utils.goodput_utils import GoodputEvent, create_goodput_recorder, maybe_monitor_goodput, maybe_record_goodput
import ray
import asyncio
import random as py_rand
import functools

from typing import Sequence
from absl import app
import jax

import max_utils
import max_logging
import pyconfig
import tensorflow as tf
import ray_cluster

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal


from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from ml_goodput_measurement import monitoring
import orbax.checkpoint as ocp

from train import (
  validate_train_config,
  train_loop
)
# pylint: disable=too-many-positional-arguments


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------

@ray.remote
class MaxtextTrainer(ray_cluster.ResilientWorker):
  def __init__(self, process_id, physical_node_id, physical_node_ip):
    super().__init__(process_id, physical_node_id, physical_node_ip)

  def initialize(self, coordinator_addr, num_processes, **kwargs):
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    tf.config.set_visible_devices([], "GPU")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    super().initialize(coordinator_addr, num_processes)
    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()
    maxtext_args = kwargs['maxtext_args']
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    self.config = pyconfig.initialize(maxtext_args)
    max_utils.print_system_information()
    validate_train_config(self.config)
    os.environ["TFDS_DATA_DIR"] = self.config.dataset_path
    self.vertex_tensorboard_manager = VertexTensorboardManager()
    if self.config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
      self.vertex_tensorboard_manager.configure_vertex_tensorboard(self.config)

    maybe_monitor_goodput(self.config)
    self.recorder = create_goodput_recorder(self.config)
      
    debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=self.config.collect_stack_trace,
          stack_trace_to_cloud=self.config.stack_trace_to_cloud,
          stack_trace_interval_seconds=self.config.stack_trace_interval_seconds,
      )
    )
    self.diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  
  def _fail(self, failure_timer_start):
    if (datetime.datetime.now() - failure_timer_start).total_seconds() >= self.config.failure_sim_time:
      if py_rand.random() >= (1 - self.config.hang_prob):
        max_logging.log("Gonna hang")
        time.sleep(3600)

      if py_rand.random() >= (1 - self.config.crash_prob):
        exception = False if py_rand.random() < 0.5 else True
        max_logging.log(f"Failing with exception = {exception}")
        if exception:
          raise Exception("Failure")
        else:
          # Cause a seg fault, no graceful exception propagation
          eval((lambda:0).__code__.replace(co_consts=()))

  def _train_loop(self, recorder, state=None):
    failure_fn = functools.partial(self._fail, failure_timer_start=datetime.datetime.now())
    train_loop(self.config, recorder, state=state, heartbeat_fn=self.heartbeat, failure_fn=failure_fn)

  def run(self):
    with diagnostic.diagnose(self.diagnostic_config):
      with maybe_record_goodput(self.recorder, GoodputEvent.JOB):
        self._train_loop(self.recorder)

def main(argv: Sequence[str]) -> None:
  ray.init(address='auto', logging_level=0)

  hang_time_threshold = None
  # Get hang time threshold
  for arg in argv:
    if arg.startswith('hang_time_threshold='):
      hang_time_threshold = int(arg.split('=')[1])
      break

  cluster_coordinator = ray_cluster.RayClusterCoordinator(MaxtextTrainer, hang_time_threshold=hang_time_threshold)
  cluster_coordinator.initialize_workers(maxtext_args=argv)
  cluster_coordinator.log("Initialized workers")
  asyncio.run(cluster_coordinator.run())



if __name__ == "__main__":
  app.run(main)

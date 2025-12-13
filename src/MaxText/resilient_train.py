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
import time
from MaxText.utils.goodput_utils import GoodputEvent, maybe_record_goodput
from MaxText import checkpointing
import ray
import asyncio
import random as py_rand
import functools

from typing import Sequence
from absl import app

import max_logging
import ray_cluster

from cloud_tpu_diagnostics import diagnostic

import orbax.checkpoint as ocp

from train import (
  initialize,
  train_loop
)

ORIGINAL_CHECKPOINTER = checkpointing.maybe_save_checkpoint

@ray.remote
class MaxtextTrainer(ray_cluster.ResilientWorker):
  def __init__(self, process_id, physical_node_id, physical_node_ip):
    super().__init__(process_id, physical_node_id, physical_node_ip)

  def initialize(self, coordinator_addr, num_processes, **kwargs):
    super().initialize(coordinator_addr, num_processes)
    maxtext_args = kwargs['maxtext_args']
    self.config, self.recorder, self.diagnostic_config = initialize(maxtext_args)
    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()
    original_checkpointer = ORIGINAL_CHECKPOINTER
    failure_fn = functools.partial(self._fail, failure_timer_start=datetime.datetime.now())
    def patched_checkpointer(*args, **kwargs):
      original_checkpointer(*args, **kwargs)
      self.heartbeat()
      failure_fn()
    checkpointing.maybe_save_checkpoint = patched_checkpointer
    
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

  def run(self):
    with diagnostic.diagnose(self.diagnostic_config):
      with maybe_record_goodput(self.recorder, GoodputEvent.JOB):
        train_loop(self.config, self.recorder, state=None)

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

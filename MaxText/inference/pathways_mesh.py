
# print("starting workload")
# import pathwaysutils
# import jax 
# from jax.sharding import PartitionSpec as P
# from jax.sharding import Mesh
# from jax.experimental import mesh_utils
# import time
# import threading
# import os 
# import signal
# import traceback


# pathwaysutils.initialize()
# print("pathwaysutils initialized")

# devices = jax.devices()

# mesh1 = mesh_utils.create_device_mesh([2, len(devices)//4], devices[:len(devices)//2])
# mesh1 = Mesh(mesh1, ('x', 'y'))

# mesh2 = mesh_utils.create_device_mesh([2, len(devices)//4], devices[len(devices)//2:]) 
# mesh2 = Mesh(mesh2, ('x', 'y'))

# print("mesh created")
# print(mesh1)
# print(mesh2)

# spec = P('x', 'y')

# sharding1 = jax.sharding.NamedSharding(mesh1, P('x', 'y'))
# sharding2 = jax.sharding.NamedSharding(mesh2, P('x', 'y'))

# def worker(weights):
#     time.sleep(10)
#     return weights+1

# shardings = [sharding1, sharding2]

# jitted_workers = [jax.jit(
#       worker,
#       in_shardings=sharding,
#       out_shardings=sharding,
#     ) for sharding in shardings]


# weights_mesh1 = jax.device_put(jax.numpy.zeros((128, 128)), sharding1)
# weights_mesh2 = jax.device_put(jax.numpy.zeros((128, 128)), sharding2)


# # Thread one runs worker() every 2 seconds 
# # Thread two runs worker() every 3 seconds
# # launch threads 

# class JetThread(threading.Thread):
#   """Thread class with exception handling to prevent silent failures."""

#   def run(self):
#     try:
#       super().run()
#     except Exception as e:  # pylint: disable=broad-exception-caught
#       traceback.print_exc()
#       # Kill the process if a thread encounters an error
#       os.kill(os.getpid(), signal.SIGKILL)

# def thread_one():
#     for _ in range(3):
#         jitted_workers[0](weights_mesh1)
#         time.sleep(5)
#     print("thread one finished")

# def thread_two():
#     for _ in range(3):
#         jitted_workers[1](weights_mesh2)
#         time.sleep(5)
#     print("thread two finished")

# thread_one = JetThread(target=thread_one)
# thread_two = JetThread(target=thread_two)

# start_time = time.time()
# thread_one.start()
# thread_two.start()

# thread_one.join()
# thread_two.join()
# end_time = time.time()
# print("all threads finished")
# print("time taken:", end_time-start_time)



# # print("start time:", time.time())
# # start_time = time.time()
# # for jitted_worker in jitted_workers:
# #     jitted_worker(weights_mesh1)
# #     print("finished first worker")
# #     jitted_worker(weights_mesh2)
# #     print("finished second worker")
# # print("end time:", time.time()-start_time)

# #     # devices_array = max_utils.create_device_mesh(config)
# #     # flat_devices = devices_array.flatten()
# #     # num_inference_devices = config.inference_devices_per_replica * config.inference_replicas
# #     # training_devices = flat_devices[num_inference_devices:].reshape(config.ici_parallelism)
# #     # max_logging.log(f"Training: Num_devices: {jax.device_count() - num_inference_devices}, shape {training_devices.shape}")
# #     # mesh = Mesh(training_devices, config.mesh_axes)
# #     # inference_devices = flat_devices[:num_inference_devices].reshape((config.inference_replicas, config.inference_devices_per_replica))
# #     # inference_meshes = [Mesh(devices.reshape(config_inference.ici_parallelism), config_inference.mesh_axes) for devices in inference_devices]
# #     # max_logging.log(f"Inference: Num_devices: {num_inference_devices}, replicas: {config.inference_replicas} with shape {tuple(inference_meshes[0].shape.values())}")

# # #   engines = [maxengine.MaxEngine(config_inference, devices=np.squeeze(inference_mesh.devices)) for inference_mesh in inference_meshes]
# # #   _ = [engine.load_params(rng_load_params) for engine in engines]


# Copyright 2024 Google LLC
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

import pathwaysutils
pathwaysutils.initialize()

import sys
import unittest
import os.path

from MaxText.globals import PKG_DIR
import jax
import jax.numpy as jnp
from MaxText.maxengine import MaxEngine
from MaxText import pyconfig
import random
import time

from MaxText.inference.offline_engine import OfflineEngine, InputData


def init_pyconfig(**kwargs):
  init_kwargs = {
      "per_device_batch_size": 4,
      "ici_data_parallelism": 1,
      "ici_fsdp_parallelism": 1,
      "ici_tensor_parallelism": -1,  # Use TP
      "run_name": "test",
      "attention": "dot_product",
      "max_prefill_predict_length": 128,
      "max_target_length": 256,
      # "model_name": "gemma2-2b",
      "base_emb_dim": 512,
      "base_num_query_heads": 32,
      "base_num_kv_heads": 32,
      "base_num_decoder_layers": 2,
      "skip_jax_distributed_system": True
  } | kwargs
  config = pyconfig.initialize(
      [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
      **init_kwargs,
  )
  return config

def test_no_batch_prefill():

  config = init_pyconfig(scan_layers=True)

  random.seed(42)
  # input_data = [jax.numpy.arange(random.randint(1, config.max_prefill_predict_length)) for _ in range(20)]
  input_data = [jax.numpy.arange(config.max_prefill_predict_length) for _ in range(2)]

  # engine = MaxEngine(config, jax.devices())
  inference_engine = OfflineEngine(config, params=None, enable_batch_prefill=False, using_pathways=True)

  inference_engine.warm_up()
  start_time = time.time()
  # jax.profiler.start_trace("gs://wenxindong-vm/trace/pathways")
  results = inference_engine.batch_inference(input_data, data_is_padded=True)
  # jax.profiler.stop_trace()
  end_time = time.time()
  # print("results: ", results)

  total_tokens = 0
  for i, tokens in enumerate(results):
    text = inference_engine.tokenizer.decode(tokens)
    print(text)
    total_tokens += len(tokens)

  print(f"Time taken: {end_time - start_time} seconds")
  print(f"Total tokens: {total_tokens}")
  print(f"Tokens per second: {total_tokens / (end_time - start_time)}")

test_no_batch_prefill()
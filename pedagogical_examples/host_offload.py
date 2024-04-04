#!/usr/bin/python3

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

'''
Frequently folks want to offload tensors to the CPU. As of March 2024, that is nicely supported in Jax via sharding annotations.
'''

from functools import partial

from absl import app
from absl import flags
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax._src.pjit import with_sharding_constraint

import argparse
import datetime
import numpy as np
from typing import Sequence

jax.config.update('jax_enable_memories', True)

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh_axis_name = "axis"
global_mesh = Mesh(devices, (mesh_axis_name))
array_on_device_sharding = jax.sharding.NamedSharding(global_mesh, jax.sharding.PartitionSpec("axis"))

small_data_dim = 3072
big_data_dim = 4*small_data_dim
num_tensors = 16

@partial(jax.jit, out_shardings=array_on_device_sharding )
def generate_array():
    return jax.numpy.ones( (small_data_dim, big_data_dim), dtype = jax.numpy.float32)

data = [generate_array() for i in range(num_tensors)]
shardings = jax.tree.map(lambda x : x.sharding, data)

host_out_shardings = jax.tree.map(lambda x : x.with_memory_kind('pinned_host'), shardings)
device_out_shardings = jax.tree.map(lambda x : x.with_memory_kind('device'), shardings)


@partial(jax.jit, out_shardings = host_out_shardings)
def put_to_host(x):
  return x

@partial(jax.jit, out_shardings = device_out_shardings)
def put_to_device(x):
  return x

num_iters = 10


#host_datas_warm = []
#for i in range(num_iters*2):
#   host_datas_warm.append(put_to_host(data))
#jax.block_until_ready(host_datas_warm)
#jax.tree_map(lambda x : x.delete(), host_datas_warm)

host_datas = []
jax.profiler.start_trace("gs://runner-maxtext-logs/rwitten_whatever/host_offload")
s = datetime.datetime.now()
for i in range(num_iters):
   host_datas.append(put_to_host(data))
jax.block_until_ready(host_datas)
e = datetime.datetime.now()
jax.profiler.stop_trace()

seconds = (e-s).total_seconds()
total_bandwidth_GB = num_iters*4*small_data_dim*big_data_dim*num_tensors/1e9
bandwidth_GB_per_s = (total_bandwidth_GB)/(seconds)
print(f"Total time {seconds=}, {total_bandwidth_GB=} {bandwidth_GB_per_s=}")

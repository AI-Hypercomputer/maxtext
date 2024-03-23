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

'''This script is an example collective matmul.'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map

MESH_DATA_AXIS = "dp"
MESH_FSDP_AXIS = "fsdp"
MESH_TENSOR_AXIS = "tp"

d = jax.devices()
outd = [[[d[0], d[1], d[3], d[2]]]]
global_mesh = Mesh(outd, (MESH_DATA_AXIS, MESH_FSDP_AXIS, MESH_TENSOR_AXIS))
print(global_mesh.shape)

batch_size = 2
seq_len = 8192
n_heads = 128
head_dim = 128
emb_dim = 16384

import random
import string
import datetime

def simple_timeit(f, *args, tries = 10, trace_base_dir = None, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

    if trace_base_dir:
        trace_dir = f"{trace_base_dir}/{trace_name}"
    else:
        trace_dir = None

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    if trace_dir:
        jax.profiler.start_trace(trace_dir)

    for _ in range(tries):
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append(1000*(e-s).total_seconds())
    if trace_dir:
        jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    print(f"{task}: average time milliseconds: {average_time_ms:.2f}")
    return average_time_ms

# gen data
def gen_data_fn():
    key = jax.random.PRNGKey(np.random.randint(0, 256))
    activations = jax.random.normal(key, shape=(batch_size, seq_len, emb_dim), dtype=jnp.bfloat16)
    weights = jax.random.normal(key, shape=(emb_dim, n_heads, head_dim), dtype=jnp.bfloat16)
    return activations, weights

data_fn = pjit(
    gen_data_fn,
    out_shardings=(P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None), P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None)),
)

def matmul(activations, weights):
    return jnp.einsum("bsE,Ehd->bshd", activations, weights)

jit_matmul = pjit(matmul, out_shardings=P(MESH_FSDP_AXIS, None, MESH_TENSOR_AXIS, None))

@partial(
    shard_map,
    mesh=global_mesh,
    in_specs=(
        P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None),
        P(MESH_FSDP_AXIS, MESH_TENSOR_AXIS, None),
    ),
    out_specs=P(MESH_FSDP_AXIS, None, MESH_TENSOR_AXIS, None),
    check_rep=False,
)
def collective_matmul(activations, weights):
    print(f"sh_map {activations.shape=} {weights.shape=}")

    axis_size = jax.lax.psum(1, axis_name=MESH_TENSOR_AXIS)
    axis_index = jax.lax.axis_index(axis_name=MESH_TENSOR_AXIS)
    # The current sequence chunk
    chunk_size = activations.shape[1]
    mid_chunk = chunk_size // 2
    # create accum buffer
    accum = jnp.zeros(
        (
            activations.shape[0],
            activations.shape[1] * axis_size,
            weights.shape[-2],
            weights.shape[-1],
        ),
        dtype=activations.dtype,
    )

    # compute first chunk
    update = jnp.einsum("bsE,Ehd->bshd", activations, weights)
    update_index = (0, axis_index * chunk_size, 0, 0)
    accum = jax.lax.dynamic_update_slice(accum, update, update_index)
    activation_forward, activation_backward = jnp.split(activations, 2, axis=1)
    activation_forward = jax.lax.ppermute(
        activation_forward,
        axis_name=MESH_TENSOR_AXIS,
        perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
    )
    activation_backward = jax.lax.ppermute(
        activation_backward,
        axis_name=MESH_TENSOR_AXIS,
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
    )

    # split activations into chunks and send
    def scanned_call(i, carrys):
        accum, activation_forward, activation_backward = carrys
        update_forward = jnp.einsum("bsE,Ehd->bshd", activation_forward, weights)
        update_backward = jnp.einsum("bsE,Ehd->bshd", activation_backward, weights)

        activation_forward = jax.lax.ppermute(
            activation_forward,
            axis_name=MESH_TENSOR_AXIS,
            perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
        )
        activation_backward = jax.lax.ppermute(
            activation_backward,
            axis_name=MESH_TENSOR_AXIS,
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        forward_update_index = ((axis_index - i - 1) % axis_size) * chunk_size
        backward_update_index = ((axis_index + i + 1) % axis_size) * chunk_size + mid_chunk

        accum = jax.lax.dynamic_update_slice(accum, update_forward, (0, forward_update_index, 0, 0))
        accum = jax.lax.dynamic_update_slice(accum, update_backward, (0, backward_update_index, 0, 0))
        return (accum, activation_forward, activation_backward)
    
    print(f"{accum.shape=}")

    accum, _, _ = jax.lax.fori_loop(
        0, (axis_size - 1), scanned_call, (accum, activation_forward, activation_backward)
    )
    return accum

with global_mesh:
    activations, weights = data_fn()

    jax.block_until_ready(activations)
    jax.block_until_ready(weights)

    @jax.jit
    def run_naive(_activations, _weights):
        with jax.named_scope("naive_matmul"):
            outputs = jit_matmul(_activations, _weights)
        return outputs

    @jax.jit
    def run_collective(_activations, _weights):
        with jax.named_scope("collective_matmul"):
            manual_outputs = jax.jit(collective_matmul)(_activations, _weights)
        return manual_outputs



    
    naive_outputs = run_naive(activations, weights)
    collective_outputs = run_collective(activations, weights)

    print(f"input {activations.shape=} {activations.addressable_shards[0].data.shape=}")
    print(f"input {weights.shape=} {weights.addressable_shards[0].data.shape=}")
    print(f"naive_outputs {naive_outputs.shape=} {naive_outputs.addressable_shards[0].data.shape=}")
    print(f"collective_outputs {collective_outputs.shape=} {collective_outputs.addressable_shards[0].data.shape=}")



    assert jnp.allclose(naive_outputs, collective_outputs), "Two algorithms should match but don't"

    simple_timeit(run_naive, activations, weights, task = "naive")
    simple_timeit(run_collective, activations, weights, task = "collective")


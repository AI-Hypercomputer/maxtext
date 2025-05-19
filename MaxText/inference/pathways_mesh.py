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

# import pathwaysutils
# pathwaysutils.initialize()

import sys
import unittest
import os.path
import jax

from MaxText.globals import PKG_DIR

import jax.numpy as jnp
from MaxText.maxengine import MaxEngine
from MaxText import pyconfig
import random
import time

import numpy as np
from jax.sharding import Mesh
from MaxText.inference.offline_engine import OfflineEngine, InputData, CompletionOutput


def get_metrics(results: list[CompletionOutput], start_time, end_time):
    total_tokens = 0
    for i, tokens in enumerate(results):
        total_tokens += len(tokens.token_ids)
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")


def init_pyconfig(**kwargs):
    init_kwargs = {
        "run_name": "test",
        # Parallelism
        "per_device_batch_size": 12,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": -1,  # Use TP
        # Inference
        "max_prefill_predict_length": 1024,
        "max_target_length": 1030,
        "return_log_prob": True,
        # Model
        "model_name": "llama2-70b",
        "attention": "dot_product",
        "scan_layers": False,
        # Quantization
        "quantization": "int8",
        "quant_cfg_path": "",
        "quantize_kvcache": True,
        "kv_quant_dtype": "int4",
        "checkpoint_is_quantized": True,
        # Base model
        # "base_emb_dim": 512,
        # "base_num_query_heads": 32,
        # "base_num_kv_heads": 32,
        # "base_num_decoder_layers": 2,
        # Checkpoints
        "tokenizer_path": "./assets/tokenizer.llama2",
        # "load_parameters_path": "gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_",
        # "skip_jax_distributed_system": True, # Single host
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config


config = init_pyconfig()
print("initialized pyconfig")
random.seed(42)

print("generating input data")
rand_l = [random.randint(1, config.max_prefill_predict_length) for _ in range(5)]

start_time = time.time()
input_data = [np.arange(l) for l in rand_l]
end_time = time.time()
print(f"Time taken to generate input data: {end_time - start_time} seconds")
# input_data = [jax.numpy.arange(config.max_prefill_predict_length) for _ in range(2)]


def test_simple_trace():
    jax.profiler.start_trace("gs://wenxindong-vm/trace/pathways/simple_trace")
    a = jax.numpy.arange(1000)
    b = jax.numpy.arange(1000)
    c = a + b
    d = jax.numpy.sum(c)
    e = jax.numpy.sum(d)
    f = jax.numpy.sum(e)
    g = jax.numpy.sum(f)
    
    jax.profiler.stop_trace()


def test_max_engine_decode():
    maxengine = MaxEngine(config)
    params=maxengine.load_params(None, rng=jax.random.PRNGKey(0))
    decode_state = maxengine.init_decode_state(rng=jax.random.PRNGKey(0))
    jax.profiler.start_trace("gs://wenxindong-vm/trace/pathways/max_engine_decode_2")
    for i in range(10):
        start_time = time.time()
        decode_state, result_tokens = maxengine.generate(params, decode_state, rng=jax.random.PRNGKey(0))
        result_tokens.data.block_until_ready()
        end_time = time.time()
        print(f"Time taken to run 1 step decode: {end_time - start_time} seconds")
    jax.profiler.stop_trace()
def test_decode():
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=False,
        rng=jax.random.PRNGKey(0),
    )

    start_time = time.time()
    inference_engine.replica_workers[0].decode()
    end_time = time.time()
    print(f"Time taken to run decode: {end_time - start_time} seconds")
    

def test_correctness():
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=False,
        rng=jax.random.PRNGKey(0),
    )
    text = "the sky is so"
    tokens, true_length = inference_engine.tokenizer.encode(text)
    print(tokens)
    input_data = [InputData(id="input_1", tokens=tokens, true_length=true_length) ]
    results = inference_engine.batch_inference(input_data)[0]
    result_tokens = results.token_ids
    detokenized_tokens = inference_engine.tokenizer.decode(result_tokens)
    print(detokenized_tokens)

    # jax.profiler.start_trace("gs://wenxindong-vm/trace/pathways/gemma2-2b-no-print")
    # results = inference_engine.batch_inference(input_data)[0]
    # jax.profiler.stop_trace()

    
    print(results)


def test_offline_engine_compare_warm_up(dp=1):
    print("Testing offline engine compare warm up")
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=dp,
        warm_up=True,
    )
    start_time1 = time.time()
    results1 = inference_engine.batch_inference(input_data)
    end_time1 = time.time()
    del inference_engine
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=dp,
        warm_up=False,
    )
    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()

    get_metrics(results1, start_time1, end_time1)
    get_metrics(results, start_time, end_time)



def test_no_dp_no_batch_prefill(
    profile=False, profile_path="gs://wenxindong-multipod-dev/trace/pathways/dp1/2"
):
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=True,
    )

    if profile:
        jax.profiler.start_trace(profile_path)

    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()

    if profile:
        jax.profiler.stop_trace()

    get_metrics(results, start_time, end_time)

def test_no_dp_batch_prefill(
    profile=False, profile_path="gs://wenxindong-multipod-dev/trace/pathways/dp1/2"
):
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=True,
        auto_layout_supported=False,
        dp=1,
        warm_up=True,
    )

    if profile:
        jax.profiler.start_trace(profile_path)

    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()

    if profile:
        jax.profiler.stop_trace()

    get_metrics(results, start_time, end_time)


def test_dp_no_batch_prefill(
    profile=False, profile_path=None
):
    start_time = time.time()
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=4,
        warm_up=False,
    )
    end_time = time.time()
    print(f"Final: Time taken to initialize engine: {end_time - start_time} seconds")
    
    if profile:
        jax.profiler.start_trace(profile_path)

    start_time = time.time()
    results: list[CompletionOutput] = inference_engine.batch_inference(input_data)
    end_time = time.time()

    if profile:
        jax.profiler.stop_trace()

    get_metrics(results, start_time, end_time)


def test_dp_batch_prefill(
    profile=False, profile_path="gs://wenxindong-multipod-dev/trace/pathways/dp2/6"
):
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=True,
        auto_layout_supported=False,
        dp=8,
        warm_up=True,
    )

    if profile:
        jax.profiler.start_trace(profile_path)

    start_time = time.time()
    results: list[CompletionOutput] = inference_engine.batch_inference(input_data)
    end_time = time.time()

    if profile:
        jax.profiler.stop_trace()

    get_metrics(results, start_time, end_time)


def test_offline_engine_init_with_params():
    maxengine = MaxEngine(config)
    params = maxengine.load_params(None, rng=jax.random.PRNGKey(0))
    
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
    assert params is not None
    inference_engine = OfflineEngine(
        config,
        params=params,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        warm_up=False, 
    )
    inference_engine.batch_inference(input_data)


def test_offline_engine_dp_init_with_params_and_dp_meshes():
    maxengine = MaxEngine(config)
    params = maxengine.params

    dp_meshes = []
    devices = jax.devices()
    num_devices_per_replica = len(devices) // 8
    mesh_shape = config.ici_parallelism

    for i in range(8):
        mesh_devices = np.array(
            devices[
                i * num_devices_per_replica : (i + 1) * num_devices_per_replica
            ]
        ).reshape(mesh_shape)
        dp_meshes.append(Mesh(mesh_devices, config.mesh_axes))

    inference_engine = OfflineEngine(
        config,
        params=params,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=8,
        dp_meshes=dp_meshes,
    )
    inference_engine.batch_inference(input_data)



def test_offline_engine_update_params():
    def reshard_params(params):
        # TODO: implement resharding
        return params

    maxengine = MaxEngine(config)
    params = maxengine.load_params(None, rng=jax.random.PRNGKey(0))

    inference_engine = OfflineEngine(
        config,
        params=params,
        enable_batch_prefill=False,
        auto_layout_supported=False,
    )

    params = reshard_params(inference_engine.params)
    inference_engine.update_params(params)

    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()
    get_metrics(results, start_time, end_time)


def test_offline_engine_input_data():
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
    )
    input_data = [InputData(id=f"input_{i}", tokens=jax.numpy.arange(config.max_prefill_predict_length)) for i in range(2)]
    results = inference_engine.batch_inference(input_data)
    print(results)


# test_simple_trace()
# test_correctness()
test_max_engine_decode()

# Time taken to run 1 step decode: 14.144538164138794 seconds
# Time taken to run 1 step decode: 2.612926721572876 seconds
# Time taken to run 1 step decode: 0.006930112838745117 seconds
# Time taken to run 1 step decode: 0.00624537467956543 seconds

# test_decode()

# time taken to run generate_fn: 12.83881950378418 seconds
# time taken to run generate_fn: 2.653505325317383 seconds
# time taken to run generate_fn: 0.006362199783325195 seconds
# time taken to run generate_fn: 0.005791902542114258 seconds
# time taken to run generate_fn: 0.0055179595947265625 seconds

# test_offline_engine_compare_warm_up(dp=1)
# test_no_dp_no_batch_prefill()
# test_no_dp_batch_prefill()
# test_dp_no_batch_prefill(profile=False, profile_path="gs://wenxindong-multipod-dev/trace/pathways/dp2/11")
# test_dp_batch_prefill()
# test_offline_engine_init_with_params()
# test_offline_engine_update_params()
# test_offline_engine_dp_init_with_params_and_dp_meshes()


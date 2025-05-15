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

import numpy as np
from jax.sharding import Mesh
from MaxText.inference.offline_engine import OfflineEngine, InputData, CompletionOutput


def get_metrics(results, start_time, end_time):
    total_tokens = 0
    for i, tokens in enumerate(results):
        total_tokens += len(tokens)
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")


def init_pyconfig(**kwargs):
    init_kwargs = {
        "run_name": "test",
        # Parallelism
        "per_device_batch_size": 4,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": -1,  # Use TP
        # Inference
        "max_prefill_predict_length": 128,
        "max_target_length": 256,
        "return_log_prob": True,
        # Model
        "model_name": "llama2-70b",
        "attention": "dot_product",
        "quantization": "int8",
        "quant_cfg_path": "",
        "checkpoint_is_quantized": True,
        "quantize_kvcache": True,
        "kv_quant_dtype": "int4",
        "scan_layers": False,
        # "base_emb_dim": 512,
        # "base_num_query_heads": 32,
        # "base_num_kv_heads": 32,
        # "base_num_decoder_layers": 2,
        # Checkpoint
        "tokenizer_path": "./assets/tokenizer.llama2",
        "load_parameters_path": "gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_",
        "skip_jax_distributed_system": True,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config


config = init_pyconfig()
random.seed(42)
# input_data = [jax.numpy.arange(random.randint(1, config.max_prefill_predict_length)) for _ in range(20)]
input_data = [jax.numpy.arange(config.max_prefill_predict_length) for _ in range(2)]


def test_offline_engine_compare_warm_up():
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=True,
    )
    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()
    get_metrics(results, start_time, end_time)

    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=False,
    )
    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()
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
    profile=False, profile_path="gs://wenxindong-multipod-dev/trace/pathways/dp2/6"
):
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
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
    params = maxengine.params

    inference_engine = OfflineEngine(
        config,
        params=params,
        enable_batch_prefill=False,
        auto_layout_supported=False,
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

    inference_engine = OfflineEngine(
        config,
        params=maxengine.params,
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

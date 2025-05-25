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
from MaxText import max_logging
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
    max_logging.log(f"Time taken: {end_time - start_time} seconds")
    max_logging.log(f"Total tokens: {total_tokens}")
    max_logging.log(f"Tokens per second: {total_tokens / (end_time - start_time)}")


def init_pyconfig(**kwargs):
    init_kwargs = {
        "run_name": "test",
        # Parallelism
        "per_device_batch_size": 4,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": -1,  # Use TP
        # Inference
        "max_prefill_predict_length": 512,
        "max_target_length": 512 + 8,
        "return_log_prob": True,
        # Model
        "model_name": "gemma2-2b",
        "attention": "dot_product",
        # Base model
        # "base_emb_dim": 512,
        # "base_num_query_heads": 32,
        # "base_num_kv_heads": 32,
        # "base_num_decoder_layers": 2,
        "scan_layers": False,
        # Quantization
        # "quantization": "int8",
        # "quant_cfg_path": "",
        # "quantize_kvcache": True,
        # "kv_quant_dtype": "int4",
        # "checkpoint_is_quantized": True,
        # Checkpoints
        "tokenizer_path": "./assets/tokenizer.llama2",
        # "load_parameters_path": "gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_",
        "skip_jax_distributed_system": True,  # Single host
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config


date = "0524"
config = init_pyconfig()
max_logging.log("initialized pyconfig")
random.seed(42)

max_logging.log("generating input data")
rand_l = [random.randint(1, config.max_prefill_predict_length) for _ in range(500)]

start_time = time.time()
input_data = [np.arange(l) for l in rand_l]
end_time = time.time()
max_logging.log(f"Time taken to generate input data: {end_time - start_time} seconds")


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


def test_max_engine_decode(
    profile=False,
    profile_path=f"gs://wenxindong-vm/trace/{date}/pathways/max_engine_decode",
):
    maxengine = MaxEngine(config)
    params = maxengine.load_params(None, rng=jax.random.PRNGKey(0))
    decode_state = maxengine.init_decode_state(rng=jax.random.PRNGKey(0))
    if profile:
        jax.profiler.start_trace(profile_path)
    result_tokens = None
    for i in range(10):
        start_time = time.time()
        decode_state, result_tokens = maxengine.generate(
            params, decode_state, rng=jax.random.PRNGKey(0)
        )
        result_tokens.data.block_until_ready()
        end_time = time.time()
        max_logging.log(
            f"Time taken to run 1 step decode: {end_time - start_time} seconds"
        )
    result_tokens.data.block_until_ready()
    if profile:
        jax.profiler.stop_trace()



def test_correctness(
    dp=1,
    num_input=1,
    num_steps=10,
    profile=False,
    profile_path=f"gs://wenxindong-multipod-dev/trace/{date}/pathways/test_correctness",
):
    inference_engine = OfflineEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=dp,
        warm_up=False,
        rng=jax.random.PRNGKey(0),
        eos_ids=[],
    )
    start_time = time.time()
    max_logging.log(f"First Round. Start time: {start_time}")
    results = inference_engine.batch_inference(input_data[:num_input])
    end_time = time.time()
    max_logging.log(
        f"First Round. Time taken to run {num_input} inference samples: {end_time - start_time} seconds"
    )

    if profile:
        jax.profiler.start_trace(profile_path)

    start_time = time.time()
    max_logging.log(f"Second Round. Start time: {start_time}")
    results = inference_engine.batch_inference(input_data[:num_input])
    end_time = time.time()
    max_logging.log(
        f"Second Round. Time taken to run {num_input} inference samples: {end_time - start_time} seconds"
    )
    tokens = 0
    for result in results:
        tokens += len(result.token_ids)
    max_logging.log(f"Tokens generated: {tokens}")
    max_logging.log(f"Tokens per second: {tokens / (end_time - start_time)}")
    if profile:
        jax.profiler.stop_trace()


test_correctness(
    dp=8,
    num_input=8,
    profile=True,
    profile_path=f"gs://wenxindong-multipod-dev/trace/{date}/pathways/test_gemma_correctness_4",
)  

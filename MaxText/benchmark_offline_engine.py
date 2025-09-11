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
import os.path
import os
import jax
import jax.numpy as jnp

from MaxText.globals import PKG_DIR
from MaxText import max_logging
from MaxText import pyconfig
import random
import time

import numpy as np
# from MaxText.inference.offline_engine import OfflineEngine, InputData, CompletionOutput
from MaxText.inference.continuous_batching_engine import ContinuousBatchingEngine, InputData, CompletionOutput


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
        "per_device_batch_size": 1,
        "ici_data_parallelism": 4, # <=== set ici_data_parallelism to dp
        "ici_tensor_parallelism": 4, # <=== set ici_data_parallelism to tp / seq parallel
        # Inference
        "max_prefill_predict_length": 128,
        "max_target_length": 132,
        "return_log_prob": True, #<=== set this to True
        # Model
        "model_name": "gemma2-2b",
        "attention": "dot_product",
        "allow_split_physical_axes": True,
        "scan_layers": False,
        # Checkpoints
        "tokenizer_type": "huggingface",
        "tokenizer_path": "google/gemma-2-2b-it",
        "hf_access_token": os.environ.get("HF_TOKEN", ""),
        "hf_path": "trl-lib/tldr",
        # "load_parameters_path": "gs://maxtext-gemma/unified/gemma2/2b/scanned/2025-09-08-18-47/0/items",
        "skip_jax_distributed_system": True,  # Single host
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "experimental", "rl", "grpo_inference.yml")],
        **init_kwargs,
    )
    return config

config = init_pyconfig()
continous_batching = True
max_logging.log("initialized pyconfig")
random.seed(42)

data = [np.arange(1,129) for _ in range(config.global_batch_size_to_train_on)]
true_lengths = jnp.array([config.max_prefill_predict_length for _ in range(config.global_batch_size_to_train_on)])
input_data = []
for i, d in enumerate(data):
    input_data.append(
        InputData(
            id=int(i),
            tokens=jnp.array(d),
            true_length=true_lengths[i],
        )
    )


def run(
    profile=False,
    profile_path="",
):
    inference_engine = ContinuousBatchingEngine(
        config,
        params=None,
        enable_batch_prefill=False,
        # continous_batching=continous_batching, 
        rng=jax.random.PRNGKey(0),
        debug=False
    )
    # max_logging.log("Starting Warmup")
    # [inference_engine.batch_inference(input_data) for _ in range(4)]
    # max_logging.log(f"Warmup Ended")
    start_time = time.time()
    max_logging.log(f"First Round. Start time: {start_time}")
    results = inference_engine.batch_inference(input_data)
    results = jax.block_until_ready(results)
    end_time = time.time()
    max_logging.log(
        f"First Round. Time taken to run {len(input_data)} inference samples: {end_time - start_time} seconds"
    )

    if profile:
        jax.profiler.start_trace(profile_path)
        start_time = time.time()
        max_logging.log(f"Second Round. Start time: {start_time}")
        results = inference_engine.batch_inference(input_data)
        results = jax.block_until_ready(results)
        end_time = time.time()
        jax.profiler.stop_trace()
        max_logging.log(
            f"Second Round. Time taken to run {len(input_data)} inference samples: {end_time - start_time} seconds"
        )
        if continous_batching:
            tokens = 0
            for result in results:
                tokens += len(result.token_ids)
        else:
            tokens = 0
            for result in results:
                tokens += jnp.count_nonzero(result.token_ids)
        max_logging.log(f"Tokens generated: {tokens}")
        max_logging.log(f"Tokens per second: {tokens / (end_time - start_time)}")
        


run(
    profile=True,
    profile_path=f"gs://runner-maxtext-logs/mohitkhatwani_offline_benchmark/app_trial/0911/{time.strftime('%Y%m%d-%H%M%S')}/",
)

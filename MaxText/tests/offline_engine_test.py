"""
Copyright 2024 Google LLC

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

""" Tests for the maxengine """

import sys
import unittest
import os.path

from MaxText.globals import PKG_DIR
import jax
import jax.numpy as jnp
from MaxText.maxengine import MaxEngine
from MaxText.inference.offline_engine import OfflineEngine, InputData
from MaxText import pyconfig
import random
import time


class OfflineEngineTest(unittest.TestCase):
  """Tests for JetStream Offline Engine.
  Run with: pytest MaxText/tests/offline_engine_test.py -s
  """

  def setUp(self):
    super().setUp()
    self.cfg = self.init_pyconfig()
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    init_kwargs = {
        "per_device_batch_size": 4,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": -1,  # Use TP
        "run_name": "test",
        "enable_checkpointing": False,
        "attention": "dot_product",
        "max_prefill_predict_length": 512,
        "max_target_length": 600,
        "base_emb_dim": 512,
        "base_num_query_heads": 8,
        "base_num_kv_heads": 8,
        "base_num_decoder_layers": 2,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config

  def test_batch_prefill(self):

    config = self.init_pyconfig(scan_layers=False)
    engine = MaxEngine(config, jax.devices())
    inference_engine = OfflineEngine(engine=engine, params=None, enable_batch_prefill=True)

    random.seed(42)
    input_data = [jax.numpy.arange(random.randint(1, config.max_prefill_predict_length)) for _ in range(20)]

    # Run inference
    start_time = time.time()
    results = inference_engine.batch_inference(input_data, data_is_padded=False)
    end_time = time.time()

    self.assertEqual(type(results), list)

    total_tokens = 0
    for i, tokens in enumerate(results):
      text = inference_engine.tokenizer.decode(tokens)
      print(text)
      self.assertEqual(type(tokens), list)
      total_tokens += len(tokens)

    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")

  def test_no_batch_prefill(self):

    config = self.init_pyconfig(scan_layers=True)

    random.seed(42)
    input_data = [jax.numpy.arange(random.randint(1, config.max_prefill_predict_length)) for _ in range(20)]

    engine = MaxEngine(config, jax.devices())
    inference_engine = OfflineEngine(engine=engine, params=None, enable_batch_prefill=False)

    start_time = time.time()
    results = inference_engine.batch_inference(input_data, data_is_padded=False)
    end_time = time.time()

    total_tokens = 0
    for i, tokens in enumerate(results):
      text = inference_engine.tokenizer.decode(tokens)
      print(text)
      total_tokens += len(tokens)

    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per second: {total_tokens / (end_time - start_time)}")

  def test_input_data(self):

    config = self.init_pyconfig(scan_layers=True)
    engine = MaxEngine(config, jax.devices())
    inference = OfflineEngine(engine=engine, params=None, enable_batch_prefill=False)

    input_data = [
        InputData(id="sample_1", tokens=jnp.array([3] * 20), true_length=20),
        InputData(id="sample_2", tokens=jnp.array([3] * 40), true_length=30),
        InputData(id="sample_3", tokens=jnp.array([3] * 60), true_length=60),
        InputData(id="sample_4", tokens=jnp.array([3] * 120), true_length=120),
    ]

    # Run inference
    results = inference.batch_inference(input_data, desc="example_run", data_is_padded=False)

    # Process results
    for id_, tokens in results.items():
      self.assertEqual(type(tokens), list)
      text = inference.tokenizer.decode(tokens)
      print(text)


if __name__ == "__main__":
  unittest.main()

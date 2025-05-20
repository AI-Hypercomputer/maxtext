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
from MaxText.inference.offline_engine import OfflineEngine, InputData, CompletionOutput
from MaxText import pyconfig
import random
import time

if os.getenv('ON_PATHWAYS', 0) == 1 or True:
  print("Initializing Pathways")
  import pathwaysutils
  pathwaysutils.initialize()
  
class OfflineEngineTest(unittest.TestCase):
  """Tests for JetStream Offline Engine.
  Run on Pathways with: ON_PATHWAYS=1 pytest MaxText/tests/offline_engine_test.py
  Run in McJAX mode with: pytest MaxText/tests/offline_engine_test.py
  """

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)


  def init_pyconfig(self, **kwargs):
      init_kwargs = {
          "run_name": "test",
          # Parallelism
          "per_device_batch_size": 1,
          "ici_data_parallelism": 1,
          "ici_fsdp_parallelism": 1,
          "ici_tensor_parallelism": -1,  # Use TP
          # Inference
          "max_prefill_predict_length": 1024,
          "max_target_length": 1030,
          # Model
          "attention": "dot_product",
          "base_emb_dim": 512,
          "base_num_query_heads": 32,
          "base_num_kv_heads": 32,
          "base_num_decoder_layers": 2,
          "scan_layers": False,
          # Quantization
          "quantization": "int8",
          "quant_cfg_path": "",
          "quantize_kvcache": True,
          "kv_quant_dtype": "int4",
          "checkpoint_is_quantized": True,
      } | kwargs
      config = pyconfig.initialize(
          [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
          **init_kwargs,
      )
      return config


  @unittest.skipIf(int(os.getenv('ON_PATHWAYS', 0)) == 0, "Skip if not on Pathways")
  def test_single_replica_on_pathways(self):
    
    config = self.init_pyconfig()
    inference_engine = OfflineEngine(
        config = config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=False,
        rng=self.rng
    )

    input_lengths = [20, 40, 50]
    input_data = [InputData(id="input_1", tokens=jnp.array([3] * length), true_length=length) for length in input_lengths]
    results = inference_engine.batch_inference(input_data)

    assert isinstance(results, list)
    assert len(results) == len(input_lengths)
    for result, length in zip(results, input_lengths):
      assert isinstance(result, CompletionOutput)
      assert isinstance(result.token_ids, jnp.ndarray)
      assert result.token_ids.shape == (length,)
      assert isinstance(result.logprobs, jnp.ndarray)
      assert result.logprobs.shape == (length,)
      result_tokens = result.token_ids
      print(f"Tokens: {result_tokens[:5]}...")  # Print first few logprobs


  @unittest.skipIf(int(os.getenv('ON_PATHWAYS', 0)) == 0, "Skip if not on Pathways")
  def test_multi_replica_on_pathways(self):
    
    config = self.init_pyconfig(skip_jax_distributed_system=True)
    inference_engine = OfflineEngine(
        config = config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=4,
        warmup=False,
        rng=self.rng
    )

    input_lengths = list(range(10, 1000, 100))
    input_data = [InputData(id="input_1", tokens=jnp.array([3] * length), true_length=length) for length in input_lengths]
    results = inference_engine.batch_inference(input_data)

    assert isinstance(results, list)
    assert len(results) == len(input_lengths)
    for result, length in zip(results, input_lengths):
      assert isinstance(result, CompletionOutput)
      assert isinstance(result.token_ids, jnp.ndarray)
      assert result.token_ids.shape == (length,)
      assert isinstance(result.logprobs, jnp.ndarray)
      assert result.logprobs.shape == (length,)
      result_tokens = result.token_ids
      print(f"Tokens: {result_tokens[:5]}...")  # Print first few logprobs


  @unittest.skipIf(int(os.getenv('ON_PATHWAYS', 0)) == 0, "Skip if not on Pathways")
  def test_multi_sampling(self):
    
    config = self.init_pyconfig(skip_jax_distributed_system=True)
    inference_engine = OfflineEngine(
        config = config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=4,
        warm_up=False,
        rng=self.rng,
        eos_ids=[]
    )

    input_data = [InputData(id=f"input_{i}", tokens=jnp.array([3] * 1024), true_length=1024) for i in range(4)]
    results = inference_engine.batch_inference(input_data)

    assert isinstance(results, list)
    assert len(results) == 4
    # Check that completion outputs are different
    for i in range(4):
      print(f"Result {i}: {results[i].token_ids[:5]}...")
      for j in range(i+1, 4):
        assert not jnp.array_equal(results[i].token_ids, results[j].token_ids)
      

  @unittest.skipIf(int(os.getenv('ON_PATHWAYS', 0)) == 0, "Skip if not on Pathways")
  def test_replica_parallelism(self):
    
    input_data = [InputData(id=f"input_{i}", tokens=jnp.array([3] * 1024), true_length=1024) for i in range(16)]

    config = self.init_pyconfig(skip_jax_distributed_system=True, max_target_length=1224)
    inference_engine = OfflineEngine(
        config = config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=1,
        warm_up=False,
        rng=self.rng,
        eos_ids=[]
    )

    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()
    single_replica_time = end_time - start_time

    del inference_engine

    inference_engine = OfflineEngine(
        config = config,
        params=None,
        enable_batch_prefill=False,
        auto_layout_supported=False,
        dp=4,
        warm_up=False,
        rng=self.rng
    )

    start_time = time.time()
    results = inference_engine.batch_inference(input_data)
    end_time = time.time()
    multi_replica_time = end_time - start_time
    print(f"Time taken with 1 replica: {single_replica_time} seconds") #53s
    print(f"Time taken with 4 replicas: {multi_replica_time} seconds")

    assert multi_replica_time < single_replica_time

if __name__ == "__main__":
  unittest.main()

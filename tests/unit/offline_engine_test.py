# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the offline inference engine."""

import sys
import unittest
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from maxtext.configs import pyconfig
from maxtext.common.gcloud_stub import is_decoupled, jetstream

pytestmark = [pytest.mark.external_serving]

config_lib, _, _, _, _ = jetstream()
if getattr(config_lib, "_IS_STUB", False):
  pytest.skip("JetStream is not installed (stubbed)", allow_module_level=True)

# Conditional import: only load when not in decoupled mode to avoid collection errors.
# offline_engine depends on prefill_packing, which requires JetStream.
if not is_decoupled():
  from maxtext.inference.offline_engine import OfflineEngine, InputData, CompletionOutput
else:
  OfflineEngine = InputData = CompletionOutput = None  # Will never be used due to external_serving marker

from tests.utils.test_helpers import get_test_config_path


class OfflineEngineTest(unittest.TestCase):
  """Tests for JetStream Offline Engine.
  Command: pytest tests/unit/offline_engine_test.py
  """

  def setUp(self):
    super().setUp()
    self.cfg = self.init_pyconfig()

  def init_pyconfig(self, **kwargs):
    """Initialize MaxText pyconfig."""
    init_kwargs = {
        "run_name": "test",
        # Parallelism
        "per_device_batch_size": 1,
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": -1,  # Use TP
        # Inference
        "max_prefill_predict_length": 512,
        "max_target_length": 512 + 10,
        "return_log_prob": True,
        "decode_sampling_strategy": "weighted",
        # Model
        "attention": "dot_product",
        "base_emb_dim": 512,
        "base_num_query_heads": 32,
        "base_num_kv_heads": 32,
        "base_num_decoder_layers": 2,
        "scan_layers": False,
        "skip_jax_distributed_system": True,
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **init_kwargs,
    )
    return config

  def test_mcjax_tp(self):

    config = self.cfg
    rng = jax.random.PRNGKey(0)
    inference_engine = OfflineEngine(config=config, params=None, enable_batch_prefill=False, rng=rng, eos_ids=[])
    input_lengths = list(range(10, 600, 100))
    input_data = [
        InputData(id=f"input_{i}", tokens=np.arange(length), true_length=length) for i, length in enumerate(input_lengths)
    ]

    results = inference_engine.batch_inference(input_data)

    completion_length = config.max_target_length - config.max_prefill_predict_length
    for result, length in zip(results, input_lengths):
      assert isinstance(result, CompletionOutput)
      assert isinstance(result.token_ids, np.ndarray)
      assert result.token_ids.shape == (length + completion_length,)
      assert isinstance(result.logprobs, np.ndarray)
      assert result.logprobs.shape == (length + completion_length,)

  def test_multi_sampling(self):
    config = self.cfg
    rng = jax.random.PRNGKey(0)
    inference_engine = OfflineEngine(config=config, params=None, enable_batch_prefill=False, rng=rng, eos_ids=[])
    rng1, rng_2 = jax.random.split(rng, 2)
    input_data = [InputData(id=f"input_{i}", tokens=jnp.arange(128), true_length=128) for i in range(4)]

    results_1 = inference_engine.batch_inference(input_data, rng=rng1)
    results_2 = inference_engine.batch_inference(input_data, rng=rng_2)

    # Check that completion outputs are different
    for i in range(4):
      assert not jnp.array_equal(results_1[i].token_ids, results_2[i].token_ids)

  def test_prng_keys_advance_during_inference(self):
    """Every decode step and every prefill must consume a fresh PRNG key.

    `InferenceWorker` and `PrefillHelper` each hold one key. That key used to be
    handed unchanged to every `generate` call in the autoregressive loop and to
    every prefill, so a non-greedy `decode_sampling_strategy` drew every token
    from the same key.

    The stored keys are asserted on rather than the sampled tokens: a stale key
    changes how successive draws are *correlated*, not necessarily their values,
    so comparing completions would not reliably detect it.
    """
    config = self.cfg
    # TODO(zhijingli): add an enable_batch_prefill=True case so that
    # BatchedPrefillProcessor.rng is covered as well.
    inference_engine = OfflineEngine(
        config=config, params=None, enable_batch_prefill=False, rng=jax.random.PRNGKey(0), eos_ids=[]
    )
    worker = inference_engine.worker
    decode_key_before = worker.rng
    prefill_key_before = worker.prefill_helper.rng

    # rng=None, so the worker keeps advancing the key it already holds.
    inference_engine.batch_inference([InputData(id="input_0", tokens=np.arange(128), true_length=128)])

    self.assertFalse(
        jnp.array_equal(worker.rng, decode_key_before),
        "the decode loop did not advance its PRNG key; every step sampled from the same key",
    )
    self.assertFalse(
        jnp.array_equal(worker.prefill_helper.rng, prefill_key_before),
        "prefill did not advance its PRNG key; every prompt sampled its first token from the same key",
    )


if __name__ == "__main__":
  unittest.main()

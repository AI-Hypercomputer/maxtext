# Copyright 2023–2026 Google LLC
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

"""Tests for PrefillProcessor's and BatchedPrefillProcessor's ahead-of-time compiled paths."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.gcloud_stub import is_decoupled

pytestmark = [pytest.mark.cpu_only, pytest.mark.external_serving]

# prefill_packing raises on import when decoupled and jetstream is stubbed.
if not is_decoupled():
  from maxtext.input_pipeline.packing.prefill_packing import BatchedPrefillProcessor, PrefillProcessor
else:
  BatchedPrefillProcessor = PrefillProcessor = None  # Never used: the external_serving marker deselects these tests.


class _FakeEngine:
  """The parts of MaxEngine that PrefillProcessor uses, as tiny JAX functions."""

  param_layouts = None
  decode_state_layouts = {"tokens": None}  # a dict, as a real MaxEngine's after aot_compile
  decode_state_shapes = {"tokens": jax.ShapeDtypeStruct((2,), jnp.int32)}

  def prefill(self, params, padded_tokens, true_length, rng, return_prompt_logp=False):
    del rng
    first_token = padded_tokens[true_length - 1] + params["offset"]
    result = {"first_token": first_token}
    if return_prompt_logp:
      result["prompt_logp"] = jnp.zeros(padded_tokens.shape, jnp.float32)
    return result, first_token

  def insert(self, prefill_result, decode_state, slot):
    return {"tokens": decode_state["tokens"].at[slot].set(prefill_result["first_token"])}


class _FakeBatchEngine(_FakeEngine):
  """The packed-prefill parts of MaxEngine that BatchedPrefillProcessor uses."""

  def prefill_concat(
      self,
      params,
      padded_tokens,
      decoder_positions,
      decoder_segment_ids,
      start_pos,
      true_lengths,
      num_prompts,
      return_prompt_logp=False,
  ):
    """The first token of each packed prompt, and its prompt_logp when asked for."""
    del decoder_positions, decoder_segment_ids
    first_tokens = (padded_tokens[start_pos + true_lengths - 1] + params["offset"])[:num_prompts]
    prefix = {"first_tokens": first_tokens}
    if return_prompt_logp:
      prefix["prompt_logp"] = jnp.zeros((1, padded_tokens.shape[0]), jnp.float32)
    return jnp.zeros(()), prefix, first_tokens

  def insert_partial(self, prefix, decode_state, cache, slots, start_indices, num_prompts, seq_len):
    del cache, start_indices, seq_len
    return {"tokens": decode_state["tokens"].at[slots[:num_prompts]].set(prefix["first_tokens"])}


class PrefillProcessorAotTest(unittest.TestCase):
  """process() after aot_compile, as the MLPerf offline harness calls it.
  Command: pytest tests/unit/prefill_processor_test.py
  """

  def setUp(self):
    super().setUp()
    self.params = {"offset": jnp.int32(100)}
    self.rng = jnp.zeros((4,), jnp.uint32)
    self.tokens = np.zeros(8, np.int32)
    self.tokens[:3] = [5, 6, 7]

  def decode_state(self):
    return {"tokens": jnp.zeros((2,), jnp.int32)}

  def test_process_after_aot_compile(self):
    processor = PrefillProcessor(_FakeEngine())
    processor.aot_compile(self.params, len(self.tokens))
    first_token, decode_state = processor.process(
        self.params, self.decode_state(), 1, jnp.asarray(self.tokens), 3, self.rng
    )
    self.assertEqual(int(first_token), 107)
    self.assertEqual(decode_state["tokens"].tolist(), [0, 107])

  def test_return_prompt_logp_is_compiled_separately(self):
    processor = PrefillProcessor(_FakeEngine())
    processor.aot_compile(self.params, len(self.tokens))
    _, decode_state = processor.process(
        self.params, self.decode_state(), 0, jnp.asarray(self.tokens), 3, self.rng, return_prompt_logp=True
    )
    self.assertEqual(decode_state["prompt_logp"].shape, (8,))
    self.assertEqual(set(processor.process_func), {(8, False), (8, True)})

  def test_compiled_process_is_reused(self):
    processor = PrefillProcessor(_FakeEngine())
    compiled = processor.aot_compile(self.params, len(self.tokens))
    self.assertIs(processor._process_compiled(self.params, len(self.tokens)), compiled)  # pylint: disable=protected-access


class BatchedPrefillProcessorAotTest(unittest.TestCase):
  """BatchedPrefillProcessor's compiled path (auto_layout_supported=True), used after aot_compile.
  Command: pytest tests/unit/prefill_processor_test.py
  """

  def setUp(self):
    super().setUp()
    self.params = {"offset": jnp.int32(100)}
    self.prompts = [jnp.array([5, 6, 7], jnp.int32), jnp.array([8, 9], jnp.int32)]
    self.results = []

  def prefill_done(self, prefill_result, row_ids, decode_state):
    self.results.append((prefill_result, row_ids, decode_state))

  def run_bucket(self, processor, return_prompt_logp=False):
    decode_state = {"tokens": jnp.zeros((2,), jnp.int32)}
    for slot, prompt in enumerate(self.prompts):
      processor.process(
          self.params, decode_state, slot, slot, prompt, 8, 8, self.prefill_done, return_prompt_logp=return_prompt_logp
      )
    processor.flush(self.params, decode_state, self.prefill_done, return_prompt_logp=return_prompt_logp)
    return self.results[-1]

  def test_process_after_aot_compile(self):
    processor = BatchedPrefillProcessor(_FakeBatchEngine(), max_batch_size=2)
    processor.aot_compile(self.params, 8, 8, 2)
    prefill_result, row_ids, decode_state = self.run_bucket(processor)
    self.assertEqual(row_ids, [0, 1])
    self.assertEqual([int(r.result_tokens) for r in prefill_result], [107, 109])
    self.assertEqual(decode_state["tokens"].tolist(), [107, 109])

  def test_return_prompt_logp_is_compiled_separately(self):
    processor = BatchedPrefillProcessor(_FakeBatchEngine(), max_batch_size=2)
    processor.aot_compile(self.params, 8, 8, 2)
    prefill_result, _, _ = self.run_bucket(processor, return_prompt_logp=True)
    self.assertEqual([r.prompt_logp.shape for r in prefill_result], [(1, 3), (1, 2)])
    self.assertEqual(set(processor.process_batch_func), {(8, 2, False), (8, 2, True)})

  def test_compiled_process_batch_is_reused(self):
    processor = BatchedPrefillProcessor(_FakeBatchEngine(), max_batch_size=2)
    compiled = processor.aot_compile(self.params, 8, 8, 2)
    self.assertIs(processor._process_batch_compiled(self.params, 8, 8, 2, False), compiled)  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()

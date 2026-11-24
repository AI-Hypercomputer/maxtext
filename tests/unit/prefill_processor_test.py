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

"""Tests for PrefillProcessor's ahead-of-time compiled path."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.input_pipeline.packing.prefill_packing import PrefillProcessor


class _FakeEngine:
  """The parts of MaxEngine that PrefillProcessor uses, as tiny JAX functions."""

  param_layouts = None
  decode_state_layouts = None
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


@pytest.mark.cpu_only
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


if __name__ == "__main__":
  unittest.main()

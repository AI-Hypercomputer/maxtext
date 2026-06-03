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

"""Tests for TunixMaxTextAdapter segment_ids synthesis (CPU-only)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

import jax.numpy as jnp
import numpy as np

from maxtext.integration.tunix import tunix_adapter as tunix_adapter_module
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter


pytestmark = [pytest.mark.post_training]


_STUB_MODEL_NAME = "_stub_model"


class _CallableStubBase:
  """Stub MaxText Transformer that records the kwargs it is called with."""

  def __init__(self):
    self.config = SimpleNamespace(model_name=_STUB_MODEL_NAME)
    self.captured = {}

  def __call__(self, *, decoder_input_tokens, decoder_positions, decoder_segment_ids, forced_routed_experts=None):
    self.captured["decoder_input_tokens"] = decoder_input_tokens
    self.captured["decoder_positions"] = decoder_positions
    self.captured["decoder_segment_ids"] = decoder_segment_ids
    self.captured["forced_routed_experts"] = forced_routed_experts
    # Return dummy logits shaped [B, L, V=2] so the adapter has something to forward.
    b, l = decoder_input_tokens.shape
    return jnp.zeros((b, l, 2), dtype=jnp.float32)


class TunixAdapterSegmentIdsTest(unittest.TestCase):
  """Verify the pad-id-based segment_ids synthesis path in TunixMaxTextAdapter."""

  def setUp(self):
    super().setUp()
    # Stub out VllmWeightMapping and HF_MODEL_CONFIGS so the adapter constructor
    # can run without standing up a real MaxText model or touching the HF model
    # registry. Patches are scoped to one test via addCleanup.
    weight_mapping_patcher = mock.patch.object(tunix_adapter_module, "VllmWeightMapping")
    weight_mapping_patcher.start()
    self.addCleanup(weight_mapping_patcher.stop)

    hf_configs_patcher = mock.patch.dict(
        tunix_adapter_module.HF_MODEL_CONFIGS,
        {_STUB_MODEL_NAME: SimpleNamespace(to_dict=lambda: {})},
    )
    hf_configs_patcher.start()
    self.addCleanup(hf_configs_patcher.stop)

    self.base = _CallableStubBase()

  @pytest.mark.cpu_only
  def test_synthesizes_segment_ids_when_pad_id_set_and_seg_ids_none(self):
    """pad_id is set + caller passes decoder_segment_ids=None -> adapter
    synthesizes segment_ids = (input_tokens != pad_id)."""
    pad_id = 99
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=pad_id)

    # Input mixes left-padding and right-padding (mirrors typical RL batch
    # shape where prompts are left-padded and completions are right-padded).
    input_tokens = jnp.array([[10, 11, 12, 99, 99], [99, 99, 20, 21, 22]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32), (2, 5))

    adapter(input_tokens, positions, None, None, decoder_segment_ids=None)

    seg = self.base.captured["decoder_segment_ids"]
    self.assertIsNotNone(seg, "Adapter should have synthesized segment_ids, not forwarded None.")

    expected = jnp.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(seg), np.asarray(expected))
    self.assertEqual(seg.dtype, jnp.int32)

  @pytest.mark.cpu_only
  def test_does_not_synthesize_when_pad_id_is_none(self):
    """pad_id omitted -> adapter forwards decoder_segment_ids=None unchanged
    (backward-compatibility for callers that don't set pad_id)."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=None)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]

    adapter(input_tokens, positions, None, None, decoder_segment_ids=None)

    self.assertIsNone(self.base.captured["decoder_segment_ids"])

  @pytest.mark.cpu_only
  def test_passes_through_explicit_segment_ids_unchanged(self):
    """Caller-provided decoder_segment_ids should pass through verbatim
    regardless of pad_id."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]
    explicit_seg = jnp.array([[7, 7, 7, 7, 7]], dtype=jnp.int32)

    adapter(input_tokens, positions, None, None, decoder_segment_ids=explicit_seg)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(explicit_seg))

  @pytest.mark.cpu_only
  def test_returns_logits_and_none_tuple(self):
    """Adapter's __call__ contract: return (logits, None) to match Tunix's
    expected interface."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]

    result = adapter(input_tokens, positions, None, None, decoder_segment_ids=None)

    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    logits, second = result
    self.assertIsNone(second)
    self.assertEqual(logits.shape, (1, 5, 2))


if __name__ == "__main__":
  unittest.main()

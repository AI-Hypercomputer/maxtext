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

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import common as tunix_common

from maxtext.integration.tunix import tunix_adapter as tunix_adapter_module
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter


pytestmark = [pytest.mark.post_training]


_STUB_MODEL_NAME = "_stub_model"

_PAD = 99

# A realistic RL batch. Prompt and completion are concatenated into *one*
# sequence per row, so a row is padded at both ends: the prompt is left-padded
# so prompt ends line up at the same offset, and the completion is right-padded
# out to the generation cap. Two rows with different prompt and completion
# lengths, so no test can pass by accident on a symmetric batch.
#
#   row 0:  PAD PAD  10 11  20 21  PAD PAD     prompt len 2, completion len 2
#   row 1:  PAD      12 13 14  22  PAD PAD PAD prompt len 3, completion len 1
_RL_BATCH = (
    (_PAD, _PAD, 10, 11, 20, 21, _PAD, _PAD),
    (_PAD, 12, 13, 14, 22, _PAD, _PAD, _PAD),
)
_RL_NONPAD = (
    (0, 0, 1, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 0, 0, 0),
)


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

  def test_synthesizes_segment_ids_when_pad_id_set_and_seg_ids_none(self):
    """pad_id is set + caller passes decoder_segment_ids=None -> adapter
    synthesizes segment_ids = (input_tokens != pad_id)."""
    pad_id = 99
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=pad_id)

    input_tokens = jnp.array(_RL_BATCH, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(input_tokens.shape[1], dtype=jnp.int32), input_tokens.shape)

    adapter(input_tokens, positions, None, None, decoder_segment_ids=None)

    seg = self.base.captured["decoder_segment_ids"]
    self.assertIsNotNone(seg, "Adapter should have synthesized segment_ids, not forwarded None.")

    expected = jnp.array(_RL_NONPAD, dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(seg), np.asarray(expected))
    self.assertEqual(seg.dtype, jnp.int32)

  def test_does_not_synthesize_when_pad_id_is_none(self):
    """pad_id omitted -> adapter forwards decoder_segment_ids=None unchanged
    (backward-compatibility for callers that don't set pad_id)."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=None)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]

    adapter(input_tokens, positions, None, None, decoder_segment_ids=None)

    self.assertIsNone(self.base.captured["decoder_segment_ids"])

  def test_passes_through_explicit_segment_ids_unchanged(self):
    """Caller-provided decoder_segment_ids should pass through verbatim
    regardless of pad_id."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]
    explicit_seg = jnp.array([[7, 7, 7, 7, 7]], dtype=jnp.int32)

    adapter(input_tokens, positions, None, None, decoder_segment_ids=explicit_seg)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(explicit_seg))

  def test_segment_ids_alias_maps_to_decoder_segment_ids(self):
    """Tunix passes packed segment ids under the name `segment_ids`; the
    adapter must forward them as MaxText's `decoder_segment_ids` instead of
    synthesizing a pad mask."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 20, 21]], dtype=jnp.int32)
    positions = jnp.array([[0, 1, 2, 0, 1]], dtype=jnp.int32)
    packed_seg = jnp.array([[1, 1, 1, 2, 2]], dtype=jnp.int32)

    adapter(input_tokens, positions, None, None, segment_ids=packed_seg)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(packed_seg))

  def test_segment_ids_is_named_so_the_tunix_gate_passes(self):
    """Tunix forwards packed segment ids only when the model's call signature
    names `segment_ids` (tunix.rl.common.model_call_contains). Pin the name
    itself, not just the plumbing: the gate also accepts a `**kwargs`, so a
    later refactor could keep the forwarding tests green while every packed
    row silently reverts to whole-row attention."""
    params = inspect.signature(TunixMaxTextAdapter.__call__).parameters
    self.assertIn("segment_ids", params)
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)
    self.assertTrue(tunix_common.model_call_contains(adapter, "segment_ids"))

  def test_segment_ids_alias_takes_precedence_over_decoder_segment_ids(self):
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 20, 21]], dtype=jnp.int32)
    positions = jnp.array([[0, 1, 2, 0, 1]], dtype=jnp.int32)
    packed_seg = jnp.array([[1, 1, 1, 2, 2]], dtype=jnp.int32)
    other_seg = jnp.array([[7, 7, 7, 7, 7]], dtype=jnp.int32)

    adapter(input_tokens, positions, None, None, decoder_segment_ids=other_seg, segment_ids=packed_seg)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(packed_seg))

  def test_forwards_forced_routed_experts(self):
    """The stub captures the kwarg but nothing asserted on it, so deleting the

    forwarding from the adapter would have passed CI.
    """
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)

    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]
    forced = jnp.zeros((1, 5, 2), dtype=jnp.int32)

    adapter(input_tokens, positions, None, None, forced_routed_experts=forced)

    np.testing.assert_array_equal(
        np.asarray(self.base.captured["forced_routed_experts"]),
        np.asarray(forced),
    )

  def test_omits_forced_routed_experts_by_default(self):
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=99)
    input_tokens = jnp.array([[10, 11, 12, 99, 99]], dtype=jnp.int32)
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]

    adapter(input_tokens, positions, None, None)

    self.assertIsNone(self.base.captured["forced_routed_experts"])

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


def _tunix_causal_mask(input_mask):
  """Builds the [B, L, L] mask the way Tunix's `make_causal_attn_mask` does.

  Key side only -- `input_mask[..., None, :] * tril` -- which is the whole reason
  the adapter can recover `input_mask` from the mask's last query row.
  """
  seq_len = input_mask.shape[-1]
  causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
  return input_mask[:, None, :].astype(bool) & causal[None, :, :]


class TunixAdapterAttentionMaskTest(unittest.TestCase):
  """Segment ids come from Tunix's `attention_mask` in preference to the adapter's `pad_id`.

  The setUp patching is duplicated from `TunixAdapterSegmentIdsTest` rather than
  factored into a shared base, deliberately: this class is appended, so it does not
  touch any line upstream also edits.
  """

  def setUp(self):
    super().setUp()
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
    self.input_tokens = jnp.array(_RL_BATCH, dtype=jnp.int32)
    self.positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8))
    self.nonpad = jnp.array(_RL_NONPAD, dtype=jnp.int32)

  def test_mask_becomes_segment_ids(self):
    """The ordinary unpacked case: the mask's last query row is the pad mask."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)

    adapter(self.input_tokens, self.positions, None, _tunix_causal_mask(self.nonpad.astype(bool)))

    seg = self.base.captured["decoder_segment_ids"]
    np.testing.assert_array_equal(np.asarray(seg), np.asarray(self.nonpad))
    self.assertEqual(seg.dtype, jnp.int32)

  def test_mask_beats_pad_id(self):
    """The case the change exists for: Tunix's mask disagrees with `pad_id` -- it
    counts one more token as real at the left of each row, as it would if the pad
    id also occurs as a real token, or if the adapter were handed a stale one.
    The mask takes priority, which is what keeps the two sources from disagreeing."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)
    tunix_says = jnp.array([[0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)

    adapter(self.input_tokens, self.positions, None, _tunix_causal_mask(tunix_says.astype(bool)))

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(tunix_says))

  def test_mask_is_used_when_pad_id_is_none(self):
    """Previously this batch reached MaxText with no segment ids at all, so every
    pad position was attended to as if it were real."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=None)

    adapter(self.input_tokens, self.positions, None, _tunix_causal_mask(self.nonpad.astype(bool)))

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(self.nonpad))

  def test_explicit_segment_ids_beat_the_mask(self):
    """Packing sends `attention_mask=None`, but priority has to be pinned anyway:
    a caller sending both must get the segment ids, not a pad mask."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)
    mask = _tunix_causal_mask(jnp.ones((2, 8), dtype=bool))
    packed_seg = jnp.array([[1, 1, 1, 2, 2, 2, 0, 0], [1, 1, 2, 2, 3, 3, 0, 0]], dtype=jnp.int32)

    adapter(self.input_tokens, self.positions, None, mask, segment_ids=packed_seg)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(packed_seg))

  def test_pad_id_still_applies_when_no_mask_is_given(self):
    """`attention_mask=None` is what packing passes; the fallback must survive."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)

    adapter(self.input_tokens, self.positions, None, None)

    np.testing.assert_array_equal(np.asarray(self.base.captured["decoder_segment_ids"]), np.asarray(self.nonpad))

  def test_recovery_is_exact_for_every_query_row(self):
    """Guards the rank-1 assumption itself rather than one hand-picked example.
    Deliberately *not* a realistic RL layout: the holes are interior and ragged,
    so if Tunix ever applied its mask on the query side too, the last query row
    would no longer carry the full mask and this would fail."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=None)
    input_mask = jnp.array(
        [[True, True, False, True, False, True, True, False], [False, True, True, False, True, True, False, True]]
    )
    tokens = jnp.zeros((2, 8), dtype=jnp.int32)

    adapter(tokens, self.positions, None, _tunix_causal_mask(input_mask))

    np.testing.assert_array_equal(
        np.asarray(self.base.captured["decoder_segment_ids"]),
        np.asarray(input_mask.astype(jnp.int32)),
    )

  def test_wrong_rank_raises(self):
    """Shapes are static, so this raises under jit too -- see the jit test below."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)

    with self.assertRaisesRegex(ValueError, r"attention_mask has shape \(2, 8\), expected \(2, 8, 8\)"):
      adapter(self.input_tokens, self.positions, None, jnp.ones((2, 8), dtype=bool))

  def test_wrong_rank_raises_under_jit(self):
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)

    with self.assertRaisesRegex(ValueError, "attention_mask has shape"):
      jax.jit(lambda t, p, m: adapter(t, p, None, m))(self.input_tokens, self.positions, jnp.ones((2, 8), dtype=bool))

  def test_mask_survives_jit(self):
    """The extraction is a slice, so unlike the old value check it works on tracers."""
    adapter = TunixMaxTextAdapter(base_model=self.base, pad_id=_PAD)
    mask = _tunix_causal_mask(self.nonpad.astype(bool))

    # Asserting on `self.base.captured` here would read a leaked tracer, so this
    # pins only that the slice traces at all; the value is pinned eagerly above.
    logits, _ = jax.jit(lambda t, p, m: adapter(t, p, None, m))(self.input_tokens, self.positions, mask)

    self.assertEqual(logits.shape, (2, 8, 2))


if __name__ == "__main__":
  unittest.main()

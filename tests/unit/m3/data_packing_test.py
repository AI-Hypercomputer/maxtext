# Copyright 2026 Google LLC
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

"""Unit tests for Workstream 2: zero-copy 6-key `M3Batch` adapter and pre-shift document bin-packing."""

from absl.testing import absltest
import numpy as np

from maxtext.m3.core.data import (
    M3PackedDataIterator,
    adapt_dict_to_m3_batch,
    adapt_iterator_to_m3_batch,
    pack_prepared_documents,
    prepare_document_for_packing,
)


class _FakeStateful6KeyIterator:
  """Minimal 6-key iterator with `get_state` / `set_state` to test zero-copy adapter forwarding."""

  def __init__(self, batches):
    self._batches = list(batches)
    self._idx = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self._idx >= len(self._batches):
      raise StopIteration
    b = self._batches[self._idx]
    self._idx += 1
    return b

  def get_state(self):
    return {"idx": self._idx}

  def set_state(self, state):
    self._idx = int(state["idx"])


class DataPackingTest(absltest.TestCase):
  """Tests zero-copy 6-key adaptation, pre-shift EOS/SFT loss masking, M-RoPE offsets, and iterator state."""

  def test_zero_copy_6key_adapter_reuses_buffers_and_forwards_state(self):
    inputs = np.array([[10, 11, 12, 0]], dtype=np.int32)
    targets = np.array([[11, 12, 99, 0]], dtype=np.int32)
    in_pos = np.array([[0, 1, 2, 0]], dtype=np.int32)
    in_seg = np.array([[1, 1, 1, 0]], dtype=np.int32)
    tgt_seg = np.array([[1, 1, 1, 0]], dtype=np.int32)
    pixel_values = np.ones((1, 4, 8), dtype=np.float32)

    raw_batch = {
        "inputs": inputs,
        "targets": targets,
        "inputs_position": in_pos,
        "targets_position": in_pos,
        "inputs_segmentation": in_seg,
        "targets_segmentation": tgt_seg,
        "pixel_values": pixel_values,
    }
    adapted = adapt_dict_to_m3_batch(raw_batch)
    self.assertIs(adapted["inputs"], inputs)
    self.assertIs(adapted["targets"], targets)
    self.assertIs(adapted["positions"], in_pos)
    self.assertIs(adapted["segment_ids"], in_seg)
    np.testing.assert_array_equal(adapted["weights"], np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32))
    self.assertIs(adapted["extras"]["pixel_values"], pixel_values)

    wrapped = adapt_iterator_to_m3_batch(_FakeStateful6KeyIterator([raw_batch, raw_batch]))
    _ = next(wrapped)
    saved = wrapped.get_state()
    self.assertEqual(saved, {"idx": 1})
    wrapped.set_state({"idx": 0})
    self.assertEqual(wrapped.get_state(), {"idx": 0})

  def test_preshift_packing_prevents_cross_doc_bleed_and_preserves_eos_when_pad_equals_eos(self):
    eos_id = 0  # Deliberately identical to pad_id=0!
    # Doc 1: prompt [10, 11] (loss_mask=0), response [12, eos=0] (loss_mask=1) -> 3 causal pairs
    doc1 = prepare_document_for_packing(
        tokens=[10, 11, 12, eos_id],
        loss_mask=[0.0, 0.0, 1.0, 1.0],
    )
    # Doc 2: [20, 21, eos=0] -> 2 causal pairs
    doc2 = prepare_document_for_packing(
        tokens=[20, 21, eos_id],
        loss_mask=[1.0, 1.0, 1.0],
    )
    packed = pack_prepared_documents([doc1, doc2], seq_len=8, strategy="first_fit", pad_id=eos_id)
    self.assertLen(packed, 1)
    row = packed[0]

    # Positions 0..2 = doc1; positions 3..4 = doc2; positions 5..7 = padding
    np.testing.assert_array_equal(row["inputs"], [10, 11, 12, 20, 21, 0, 0, 0])
    # Notice doc1's last target at index 2 is eos_id (0), NOT doc2's first token (20)!
    np.testing.assert_array_equal(row["targets"], [11, 12, 0, 21, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["positions"], [0, 1, 2, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(row["segment_ids"], [1, 1, 1, 2, 2, 0, 0, 0])
    # Terminal EOS at index 2 and index 4 retain weight=1.0 even though target == pad_id == 0,
    # while SFT prompt target at index 0 and trailing pad slots 5..7 have weight=0.0!
    np.testing.assert_array_equal(row["weights"], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

  def test_best_fit_and_3d_mrope_positions(self):
    img_tok = 151655
    # 1 image of grid (1, 4, 4) with merge=2 -> 1 * 2 * 2 = 4 image tokens
    doc = prepare_document_for_packing(
        tokens=[5, img_tok, img_tok, img_tok, img_tok, 6, 7],
        grid_thw=np.array([[1, 4, 4]], dtype=np.int32),
        image_token_id=img_tok,
        spatial_merge_size=2,
    )
    self.assertEqual(doc.positions.shape, (6, 3))
    packed = pack_prepared_documents([doc], seq_len=8, strategy="best_fit")
    self.assertEqual(packed[0]["positions"].shape, (8, 3))

  def test_packed_data_iterator_checkpoint_roundtrip_bit_identical(self):
    docs = [[10 + i, 20 + i, 30 + i, 1] for i in range(12)]
    it1 = M3PackedDataIterator(docs, batch_size=2, seq_len=8, strategy="best_fit", seed=7)
    _ = next(it1)
    _ = next(it1)
    state = it1.get_state()
    expected_next = next(it1)

    it2 = M3PackedDataIterator(docs, batch_size=2, seq_len=8, strategy="best_fit", seed=7)
    it2.set_state(state)
    resumed_next = next(it2)
    for k in ("inputs", "targets", "positions", "segment_ids", "weights"):
      np.testing.assert_array_equal(np.asarray(expected_next[k]), np.asarray(resumed_next[k]))


if __name__ == "__main__":
  absltest.main()

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

"""Unit tests for MLPerf DS v3 continuous stream chunking in tfds_data_processing_c4_mlperf."""

import types
import unittest
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tfds = pytest.importorskip("tensorflow_datasets")

from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import (
    _pad_to_batch_size,
    chunk_token_stream,
    format_continuous_stream_fn,
    format_fn,
    preprocess_eval_dataset,
    preprocess_train_dataset,
    reduce_concat_tokens,
)


class TfdsC4MlperfStreamChunkingTest(unittest.TestCase):

  def _make_dataset_from_docs(self, docs):
    """Creates a tf.data.Dataset from a list of variable-length integer token lists."""

    def gen():
      for d in docs:
        yield {"targets": tf.constant(d, dtype=tf.int32)}

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={"targets": tf.TensorSpec(shape=[None], dtype=tf.int32)},
    )

  def test_chunk_token_stream_preserves_token_zero_and_chunks_perfectly(self):
    """Verifies that chunk_token_stream appends eod_id delimiter between documents and chunks perfectly.

    Given 3 docs:
      Doc 1: [0, 10, 20, 0, 30]
      Doc 2: [40, 0, 50, 60]
      Doc 3: [70, 80, 0]
    With eod_id=-1 and sequence_length=4:
      Document stream with delimiters: [0, 10, 20, 0, 30, -1, 40, 0, 50, 60, -1, 70, 80, 0, -1]
      Chunks produced (drop_remainder=True):
        Chunk 0: [0, 10, 20, 0]
        Chunk 1: [30, -1, 40, 0]
        Chunk 2: [50, 60, -1, 70]
    """
    docs = [
        [0, 10, 20, 0, 30],  # len 5
        [40, 0, 50, 60],  # len 4
        [70, 80, 0],  # len 3
    ]
    dataset = self._make_dataset_from_docs(docs)

    seq_len = 4
    eod_id = -1
    chunked_ds = chunk_token_stream(dataset, feature_key="targets", sequence_length=seq_len, eod_id=eod_id)
    chunks = list(chunked_ds.as_numpy_iterator())

    self.assertEqual(len(chunks), 3)
    np.testing.assert_array_equal(chunks[0]["targets"], np.array([0, 10, 20, 0], dtype=np.int32))
    np.testing.assert_array_equal(chunks[1]["targets"], np.array([30, -1, 40, 0], dtype=np.int32))
    np.testing.assert_array_equal(chunks[2]["targets"], np.array([50, 60, -1, 70], dtype=np.int32))

  def test_reduce_concat_tokens_drops_token_zero(self):
    """Demonstrates why reduce_concat_tokens cannot be reused: it drops token ID 0."""
    docs = [
        [0, 10, 20, 0, 30],
        [40, 0, 50],
    ]
    dataset = self._make_dataset_from_docs(docs)
    reduced_ds = reduce_concat_tokens(dataset, feature_key="targets", batch_size=2)
    output = list(reduced_ds.as_numpy_iterator())

    flattened = np.concatenate([o["targets"] for o in output])
    # Token 0 is dropped by boolean_mask(tokens, tf.cast(tokens, tf.bool))
    self.assertNotIn(0, flattened)
    self.assertEqual(len(flattened), 5)  # original was 8 tokens, 3 zeros were deleted

  def test_format_continuous_stream_fn_monotonic_positions_and_loss_mask(self):
    """Verifies monotonic position IDs, 0% padding, and 100% loss participation."""
    seq_len = 8
    eod_id = 99
    chunk = {"targets": tf.constant([10, 20, 30, 40, 50, 60, 70, 80], dtype=tf.int32)}
    formatted = format_continuous_stream_fn(chunk, max_target_length=seq_len, eod_id=eod_id)

    # 1. Inputs equal raw targets
    np.testing.assert_array_equal(formatted["inputs"].numpy(), np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32))

    # 2. Targets are shifted left with eod_id at the end
    np.testing.assert_array_equal(
        formatted["targets"].numpy(), np.array([20, 30, 40, 50, 60, 70, 80, 99], dtype=np.int32)
    )

    # 3. Position IDs count monotonically 0..seq_len-1
    np.testing.assert_array_equal(formatted["inputs_position"].numpy(), np.arange(seq_len, dtype=np.int32))
    np.testing.assert_array_equal(formatted["targets_position"].numpy(), np.arange(seq_len, dtype=np.int32))

    # 4. Segmentation IDs are uniform 1s (no cross-document boundary isolation)
    np.testing.assert_array_equal(formatted["inputs_segmentation"].numpy(), np.ones(seq_len, dtype=np.int32))
    np.testing.assert_array_equal(formatted["targets_segmentation"].numpy(), np.ones(seq_len, dtype=np.int32))

    # 5. Loss participation: all tokens participate (eod_mask_loss=False)
    loss_mask = formatted["targets_segmentation"].numpy() != 0
    self.assertEqual(loss_mask.sum(), seq_len)

  def test_attention_mask_invariance(self):
    """Verifies that uniform segment IDs allow full causal attention with no cross-document masking."""
    seq_len = 8
    # Uniform 1s as produced by format_continuous_stream_fn
    decoder_segment_ids = tf.ones([1, seq_len], dtype=tf.int32)

    # Pairwise attention mask logic from attention_op.py:
    # mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
    mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
    self.assertTrue(tf.reduce_all(mask).numpy())

  def test_preprocess_train_dataset_tokenized_stream_end_to_end(self):
    """Verifies end-to-end preprocess_train_dataset with is_tokenized_dataset=True."""
    # 4 documents: 9, 7, 7, 5 tokens. With 4 eod_id delimiters appended, total = 32 tokens.
    docs = [
        list(range(0, 9)),  # 9 tokens
        list(range(9, 16)),  # 7 tokens
        list(range(16, 23)),  # 7 tokens
        list(range(23, 28)),  # 5 tokens
    ]
    dataset = self._make_dataset_from_docs(docs)

    batch_size = 2
    seq_len = 8
    mock_tokenizer = types.SimpleNamespace(pad_id=-1, eos_id=1)
    processed_ds = preprocess_train_dataset(
        train_ds=dataset,
        sp_tokenizer=mock_tokenizer,
        train_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        shuffle_buffer_size=4,
        data_shuffle_seed=42,
        is_tokenized_dataset=True,
    )

    batches = list(processed_ds.as_numpy_iterator())
    # 32 tokens total // (batch_size 2 * seq_len 8 = 16 tokens/batch) = 2 batches
    self.assertEqual(len(batches), 2)

    for b in batches:
      self.assertEqual(b["inputs"].shape, (batch_size, seq_len))
      self.assertEqual(b["targets"].shape, (batch_size, seq_len))
      self.assertEqual(b["inputs_position"].shape, (batch_size, seq_len))
      self.assertEqual(b["targets_position"].shape, (batch_size, seq_len))
      self.assertEqual(b["inputs_segmentation"].shape, (batch_size, seq_len))
      self.assertEqual(b["targets_segmentation"].shape, (batch_size, seq_len))

      # Verify monotonic positions
      for i in range(batch_size):
        np.testing.assert_array_equal(b["inputs_position"][i], np.arange(seq_len))
        np.testing.assert_array_equal(b["targets_position"][i], np.arange(seq_len))

      # Verify uniform segmentation (all 1s, 0% padding)
      self.assertTrue((b["inputs_segmentation"] == 1).all())
      self.assertTrue((b["targets_segmentation"] == 1).all())

      # Verify 100% loss participation
      self.assertEqual((b["targets_segmentation"] != 0).sum(), batch_size * seq_len)

  def test_preprocess_eval_dataset_tokenized_stream_end_to_end(self):
    """Verifies end-to-end preprocess_eval_dataset with is_tokenized_dataset=True."""
    # 2 documents: 15 and 15 tokens. With 2 eod_id delimiters appended, total = 32 tokens.
    docs = [
        list(range(0, 15)),
        list(range(15, 30)),
    ]
    dataset = self._make_dataset_from_docs(docs)

    batch_size = 2
    seq_len = 8
    mock_tokenizer = types.SimpleNamespace(pad_id=-1, eos_id=1)
    eval_ds = preprocess_eval_dataset(
        eval_ds=dataset,
        sp_tokenizer=mock_tokenizer,
        eval_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        is_tokenized_dataset=True,
    )

    batches = list(eval_ds.as_numpy_iterator())
    self.assertGreaterEqual(len(batches), 1)
    b0 = batches[0]
    self.assertEqual(b0["inputs"].shape, (batch_size, seq_len))
    self.assertTrue((b0["inputs_segmentation"] == 1).all())
    self.assertTrue((b0["targets_segmentation"] == 1).all())
    np.testing.assert_array_equal(b0["inputs_position"][0], np.arange(seq_len))

  def test_chunk_token_stream_handles_empty_documents_gracefully(self):
    """Verifies that empty documents (length 0) in the dataset stream are handled without error."""
    docs = [
        [],
        [10, 20],
        [],
        [30, 40, 50],
        [],
        [60, 70, 80],
    ]
    dataset = self._make_dataset_from_docs(docs)
    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, feature_key="targets", sequence_length=seq_len)
    chunks = list(chunked_ds.as_numpy_iterator())

    # Total tokens = 8 -> 2 chunks of length 4
    self.assertEqual(len(chunks), 2)
    np.testing.assert_array_equal(chunks[0]["targets"], np.array([10, 20, 30, 40], dtype=np.int32))
    np.testing.assert_array_equal(chunks[1]["targets"], np.array([50, 60, 70, 80], dtype=np.int32))

  def test_chunk_token_stream_handles_documents_exceeding_sequence_length(self):
    """Verifies that documents larger than sequence_length (e.g. 10,000 tokens) chunk seamlessly."""
    long_doc = list(range(10000))
    dataset = self._make_dataset_from_docs([long_doc])
    seq_len = 4096
    chunked_ds = chunk_token_stream(dataset, feature_key="targets", sequence_length=seq_len)
    chunks = list(chunked_ds.as_numpy_iterator())

    # 10,000 tokens // 4096 = 2 chunks (8192 tokens); trailing 1808 tokens dropped by drop_remainder=True
    self.assertEqual(len(chunks), 2)
    np.testing.assert_array_equal(chunks[0]["targets"], np.arange(4096, dtype=np.int32))
    np.testing.assert_array_equal(chunks[1]["targets"], np.arange(4096, 8192, dtype=np.int32))

  def test_chunk_token_stream_drops_remainder_cleanly(self):
    """Verifies that token counts not divisible by sequence_length drop the trailing partial chunk."""
    docs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]  # 10 tokens total
    dataset = self._make_dataset_from_docs(docs)
    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, feature_key="targets", sequence_length=seq_len)
    chunks = list(chunked_ds.as_numpy_iterator())

    # 10 // 4 = 2 chunks (tokens 1..8); tokens 9 and 10 dropped
    self.assertEqual(len(chunks), 2)
    np.testing.assert_array_equal(chunks[0]["targets"], np.array([1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(chunks[1]["targets"], np.array([5, 6, 7, 8], dtype=np.int32))

  def test_format_continuous_stream_fn_preserves_llama3_quote_token(self):
    """Verifies that token ID 1 (quotation mark in Llama-3) is not masked from loss.

    In the legacy format_fn, any token matching eos_id=1 had its loss mask zeroed out,
    accidentally masking English quotation marks. format_continuous_stream_fn prevents this.
    """
    seq_len = 8
    # Tokens contain multiple 1s (representing quotation marks)
    chunk = {
        "targets": tf.constant([1, 42, 1, 99, 1, 7, 8, 1], dtype=tf.int32),
        "targets_position": tf.range(seq_len, dtype=tf.int32),
        "targets_segmentation": tf.ones([seq_len], dtype=tf.int32),
    }

    # Legacy format_fn zeros segmentation where targets == eos_id (1)
    legacy_formatted = format_fn(dict(chunk), eos_id=1, pad_id=0)
    self.assertIn(0, legacy_formatted["targets_segmentation"].numpy())

    # New continuous stream format preserves 100% loss participation (all 1s)
    continuous_formatted = format_continuous_stream_fn(chunk, max_target_length=seq_len, eod_id=1)
    np.testing.assert_array_equal(continuous_formatted["targets_segmentation"].numpy(), np.ones(seq_len, dtype=np.int32))

  def test_chunk_boundary_and_document_boundary_shifting(self):
    """Demonstrates next-token shifting behavior across document boundaries vs chunk boundaries."""
    # Document 1: [10, 20, 30] (ends with 30)
    # Document 2: [40, 50, 60] (starts with 40)
    docs = [[10, 20, 30], [40, 50, 60]]
    dataset = self._make_dataset_from_docs(docs)
    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, sequence_length=seq_len)
    formatted_ds = chunked_ds.map(lambda x: format_continuous_stream_fn(x, max_target_length=seq_len, eod_id=999))
    chunks = list(formatted_ds.as_numpy_iterator())

    c0 = chunks[0]
    # Chunk 0 contains [10, 20, 30, 40]
    np.testing.assert_array_equal(c0["inputs"], np.array([10, 20, 30, 40], dtype=np.int32))
    # Inner transition: token 30 (end of doc 1) correctly predicts token 40 (start of doc 2)
    self.assertEqual(c0["targets"][2], 40)
    # Chunk boundary transition: token 40 (end of chunk) predicts eod_id (999) because next token is in chunk 1
    self.assertEqual(c0["targets"][3], 999)

  def test_eval_pipeline_padding_masks_loss_on_padded_batches(self):
    """Verifies that _pad_to_batch_size sets targets_segmentation to 0 for padded eval examples."""
    docs = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 chunks of length 4
    dataset = self._make_dataset_from_docs(docs)
    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, sequence_length=seq_len)
    formatted_ds = chunked_ds.map(lambda x: format_continuous_stream_fn(x, max_target_length=seq_len))

    # Pad from 2 examples to 4 examples (batch size 4)
    padded_ds = _pad_to_batch_size(formatted_ds, batch_size=4, num_examples=2)
    examples = list(padded_ds.as_numpy_iterator())

    self.assertEqual(len(examples), 4)
    # Real examples have targets_segmentation == 1
    self.assertTrue((examples[0]["targets_segmentation"] == 1).all())
    self.assertTrue((examples[1]["targets_segmentation"] == 1).all())
    # Padded examples have targets_segmentation == 0 (masked from loss)
    self.assertTrue((examples[2]["targets_segmentation"] == 0).all())
    self.assertTrue((examples[3]["targets_segmentation"] == 0).all())

  def test_preprocess_train_dataset_untokenized_with_stream_chunking(self):
    """Verifies that untokenized raw text datasets support stream chunking when use_stream_chunking=True."""
    raw_texts = [
        "hello world",
        "foo bar baz",
        "maxtext stream",
    ]

    def gen():
      for t in raw_texts:
        yield {"targets": tf.constant(t, dtype=tf.string)}

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={"targets": tf.TensorSpec(shape=[], dtype=tf.string)},
    )

    class MockTokenizer:
      pad_id = 0
      eos_id = 1

      def encode(self, s):
        if isinstance(s, bytes):
          s = s.decode("utf-8")
        return [ord(c) for c in s]

    batch_size = 2
    seq_len = 8
    processed_ds = preprocess_train_dataset(
        train_ds=dataset,
        sp_tokenizer=MockTokenizer(),
        train_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        shuffle_buffer_size=4,
        data_shuffle_seed=42,
        is_tokenized_dataset=False,
        use_stream_chunking=True,
    )

    batches = list(processed_ds.as_numpy_iterator())
    self.assertGreaterEqual(len(batches), 1)
    b0 = batches[0]
    self.assertEqual(b0["inputs"].shape, (batch_size, seq_len))
    self.assertEqual(b0["targets"].shape, (batch_size, seq_len))
    # Verify monotonic position IDs and uniform 1s segmentation (stream chunking characteristics)
    np.testing.assert_array_equal(b0["inputs_position"][0], np.arange(seq_len))
    self.assertTrue((b0["inputs_segmentation"] == 1).all())
    self.assertTrue((b0["targets_segmentation"] == 1).all())

  def test_preprocess_train_dataset_tokenized_with_packing_fallback(self):
    """Verifies that tokenized dataset falls back to sequence packing when use_stream_chunking=False."""
    docs = [
        [10, 20, 30],
        [40, 50],
    ]
    dataset = self._make_dataset_from_docs(docs)
    mock_tokenizer = types.SimpleNamespace(pad_id=0, eos_id=1)
    batch_size = 1
    seq_len = 8
    processed_ds = preprocess_train_dataset(
        train_ds=dataset,
        sp_tokenizer=mock_tokenizer,
        train_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        shuffle_buffer_size=4,
        data_shuffle_seed=42,
        is_tokenized_dataset=True,
        use_stream_chunking=False,
    )
    batches = list(processed_ds.as_numpy_iterator())
    self.assertGreaterEqual(len(batches), 1)
    b0 = batches[0]
    # Under sequence packing, trailing padding has segment ID 0
    self.assertIn(0, b0["inputs_segmentation"][0])


if __name__ == "__main__":
  unittest.main()

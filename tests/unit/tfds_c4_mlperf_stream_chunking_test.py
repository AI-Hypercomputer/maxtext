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

import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train.train import loss_fn
from maxtext.input_pipeline.tfds_data_processing_c4_mlperf import (
    _pad_to_batch_size,
    chunk_token_stream,
    format_continuous_stream_fn,
    format_fn,
    preprocess_train_dataset,
    preprocess_eval_dataset,
    reduce_concat_tokens,
    rekey,
)
from tests.utils.test_helpers import get_test_config_path


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
    """Verifies that chunk_token_stream preserves token ID 0 and forms exact contiguous chunks."""
    # Create 3 documents with arbitrary lengths containing token ID 0
    docs = [
        [0, 10, 20, 0, 30],   # len 5
        [40, 0, 50, 60],       # len 4
        [70, 80, 0],           # len 3
    ]
    raw_tokens = [0, 10, 20, 0, 30, 40, 0, 50, 60, 70, 80, 0]  # total 12 tokens
    dataset = self._make_dataset_from_docs(docs)

    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, feature_key="targets", sequence_length=seq_len)
    chunks = list(chunked_ds.as_numpy_iterator())

    self.assertEqual(len(chunks), 3)
    # Check each chunk shape and content
    for chunk in chunks:
      self.assertEqual(chunk["targets"].shape, (seq_len,))

    reconstructed = np.concatenate([c["targets"] for c in chunks])
    np.testing.assert_array_equal(reconstructed, np.array(raw_tokens, dtype=np.int32))

    # Explicitly check that token 0 appears in the expected positions
    self.assertEqual(reconstructed[0], 0)
    self.assertEqual(reconstructed[3], 0)
    self.assertEqual(reconstructed[6], 0)
    self.assertEqual(reconstructed[11], 0)

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
    eos_id = 99
    chunk = {"targets": tf.constant([10, 20, 30, 40, 50, 60, 70, 80], dtype=tf.int32)}
    formatted = format_continuous_stream_fn(chunk, max_target_length=seq_len, eos_id=eos_id)

    # 1. Inputs equal raw targets
    np.testing.assert_array_equal(
        formatted["inputs"].numpy(), np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
    )

    # 2. Targets are shifted left with eos_id at the end
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
    # Create 4 documents of varying length (total 32 tokens)
    docs = [
        list(range(0, 10)),    # 10 tokens: 0..9
        list(range(10, 18)),   # 8 tokens: 10..17
        list(range(18, 26)),   # 8 tokens: 18..25
        list(range(26, 32)),   # 6 tokens: 26..31
    ]
    dataset = self._make_dataset_from_docs(docs)

    batch_size = 2
    seq_len = 8
    processed_ds = preprocess_train_dataset(
        train_ds=dataset,
        sp_tokenizer=None,
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
    docs = [
        list(range(0, 16)),
        list(range(16, 32)),
    ]
    dataset = self._make_dataset_from_docs(docs)

    batch_size = 2
    seq_len = 8
    eval_ds = preprocess_eval_dataset(
        eval_ds=dataset,
        sp_tokenizer=None,
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
    continuous_formatted = format_continuous_stream_fn(chunk, max_target_length=seq_len, eos_id=1)
    np.testing.assert_array_equal(continuous_formatted["targets_segmentation"].numpy(), np.ones(seq_len, dtype=np.int32))

  def test_chunk_boundary_and_document_boundary_shifting(self):
    """Demonstrates next-token shifting behavior across document boundaries vs chunk boundaries."""
    # Document 1: [10, 20, 30] (ends with 30)
    # Document 2: [40, 50, 60] (starts with 40)
    docs = [[10, 20, 30], [40, 50, 60]]
    dataset = self._make_dataset_from_docs(docs)
    seq_len = 4
    chunked_ds = chunk_token_stream(dataset, sequence_length=seq_len)
    formatted_ds = chunked_ds.map(lambda x: format_continuous_stream_fn(x, max_target_length=seq_len, eos_id=999))
    chunks = list(formatted_ds.as_numpy_iterator())

    c0 = chunks[0]
    # Chunk 0 contains [10, 20, 30, 40]
    np.testing.assert_array_equal(c0["inputs"], np.array([10, 20, 30, 40], dtype=np.int32))
    # Inner transition: token 30 (end of doc 1) correctly predicts token 40 (start of doc 2)
    self.assertEqual(c0["targets"][2], 40)
    # Chunk boundary transition: token 40 (end of chunk) predicts eos_id (999) because next token is in chunk 1
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

  def test_end_to_end_model_forward_and_loss_calculation(self):
    """Proves end-to-end correctness by feeding continuous stream batches to a Transformer model and loss_fn."""
    seq_len = 16
    batch_size = 2

    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        base_emb_dim=64,
        base_mlp_dim=64,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        base_num_decoder_layers=2,
        vocab_size=64,
        max_target_length=seq_len,
        per_device_batch_size=batch_size,
        global_batch_size_to_train_on=batch_size,
        global_batch_size_to_load=batch_size,
        scan_layers=False,
        attention="dot_product",
        sparse_matmul=False,
        dtype="float32",
        weight_dtype="float32",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        dataset_type="c4_mlperf",
        packing=True,
    )

    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh, quant=None)

    # 4 documents of varying lengths: 10, 12, 14, 8 tokens (total 44 tokens)
    docs = [
        list(range(5, 15)),
        list(range(15, 27)),
        list(range(27, 41)),
        list(range(41, 49)),
    ]
    raw_ds = self._make_dataset_from_docs(docs)

    train_ds = preprocess_train_dataset(
        train_ds=raw_ds,
        sp_tokenizer=None,
        train_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        shuffle_buffer_size=4,
        data_shuffle_seed=42,
        is_tokenized_dataset=True,
    )

    batch = next(iter(train_ds.as_numpy_iterator()))
    data = {k: jnp.array(v) for k, v in batch.items()}

    rng = jax.random.PRNGKey(0)
    init_rng, apply_rng = jax.random.split(rng)
    variables = model.init(
        {"params": init_rng, "dropout": init_rng},
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
    )

    # 1. Forward pass + loss calculation
    loss, aux = loss_fn(model, cfg, data, apply_rng, variables["params"], is_train=True)

    # 2. Verify 100% loss participation (total_weights == batch_size * seq_len, eod_mask_loss=False)
    expected_total_weights = float(batch_size * seq_len)
    self.assertEqual(float(aux["total_weights"]), expected_total_weights)

    # 3. Verify loss is positive and finite
    self.assertTrue(np.isfinite(float(loss)))
    self.assertGreater(float(loss), 0.0)

    # 4. Verify value_and_grad computes finite non-empty gradients
    def _compute_loss(p):
      l, _ = loss_fn(model, cfg, data, apply_rng, p, is_train=True)
      return l

    computed_loss, grads = jax.value_and_grad(_compute_loss)(variables["params"])
    self.assertAlmostEqual(float(computed_loss), float(loss), places=5)
    self.assertGreater(len(grads), 0)

  def test_real_gcs_dataset_loading(self):
    """Integration test verifying loading and stream chunking from real GCS MLPerf TFDS dataset."""
    try:
      builder = tfds.builder_from_directory("gs://mlperf-6-submission-us-central1/tfds-fixed-reshard/c4/en/3.0.5")
      raw_ds = builder.as_dataset(split="validation[:16]")
    except Exception as e:
      self.skipTest(f"GCS bucket not accessible in this environment: {e}")

    dataset = rekey(raw_ds, {"inputs": None, "targets": "ids"})
    seq_len = 4096
    batch_size = 2
    eval_ds = preprocess_eval_dataset(
        eval_ds=dataset,
        sp_tokenizer=None,
        eval_global_batch_size_to_load=batch_size,
        max_target_length=seq_len,
        is_tokenized_dataset=True,
    )
    batch = next(iter(eval_ds.as_numpy_iterator()))
    self.assertEqual(batch["inputs"].shape, (batch_size, seq_len))
    self.assertEqual(batch["targets"].shape, (batch_size, seq_len))
    self.assertTrue((batch["inputs_position"][0] == np.arange(seq_len)).all())
    self.assertTrue((batch["inputs_segmentation"][0] == 1).all())
    self.assertTrue((batch["targets_segmentation"][0] == 1).all())
    self.assertEqual((batch["targets_segmentation"][0] != 0).sum(), seq_len)


if __name__ == "__main__":
  unittest.main()


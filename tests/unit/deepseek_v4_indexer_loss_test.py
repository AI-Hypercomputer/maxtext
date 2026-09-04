# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for DeepSeek-V4 Compressed Sparse Attention (CSA) Indexer loss and training."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, DEFAULT_MASK_VALUE
from maxtext.configs import pyconfig
from maxtext.layers import attention_compressed
from maxtext.layers.attention_mla import indexer_losses
from maxtext.trainers.pre_train import train as pre_train


class _MockNnxDecoder(nnx.Module):
  """Minimal mock NNX decoder for pre_train.loss_fn tests."""

  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      encoder_images=None,
      encoder_image_masks=None,
      enable_dropout=False,
      decoder_target_tokens=None,
      decoder_target_mask=None,
  ):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks
    del enable_dropout, decoder_target_tokens, decoder_target_mask
    return jnp.zeros((*decoder_input_tokens.shape, self.vocab_size), dtype=jnp.float32)


class DeepSeekV4IndexerLossTest(unittest.TestCase):
  """Tests for DeepSeek-V4 CSA Indexer KL Divergence loss calculation and gradients."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 16
    self.base_emb_dim = 64
    self.base_num_query_heads = 4
    self.base_num_kv_heads = 1
    self.head_dim = 32
    self.compress_ratio = 4
    self.indexer_n_heads = 4
    self.indexer_head_dim = 32
    self.indexer_topk = 2
    self.q_lora_rank = 32
    self._get_config()

  def _get_config(
      self,
      use_indexer=True,
      indexer_loss_scaling_factor=0.5,
      indexer_sparse_training=False,
      mla_qk_head_chunk_size=0,
      indexer_topk=None,
  ):
    """Constructs a test MaxTextConfig with CSA indexer configuration."""
    topk = indexer_topk if indexer_topk is not None else self.indexer_topk
    argv = [
        "",
        "src/maxtext/configs/base.yml",
        "run_name=test_dsv4_indexer",
        "decoder_block=deepseek4",
        "attention_type=compressed",
        "attention=dot_product",
        f"use_indexer={use_indexer}",
        f"indexer_loss_scaling_factor={indexer_loss_scaling_factor}",
        f"indexer_sparse_training={indexer_sparse_training}",
        f"mla_qk_head_chunk_size={mla_qk_head_chunk_size}",
        f"max_target_length={self.seq_len}",
        f"indexer_topk={topk}",
        f"indexer_n_heads={self.indexer_n_heads}",
        f"indexer_head_dim={self.indexer_head_dim}",
        f"base_emb_dim={self.base_emb_dim}",
        f"base_num_query_heads={self.base_num_query_heads}",
        f"base_num_kv_heads={self.base_num_kv_heads}",
        f"head_dim={self.head_dim}",
        f"qk_rope_head_dim={self.head_dim}",
        f"q_lora_rank={self.q_lora_rank}",
        "o_groups=2",
        "o_lora_rank=16",
        "enable_checkpointing=False",
        "vocab_size=32",
    ]
    return pyconfig.initialize(argv)

  def _init_csa_attention(self, config):
    """Initializes a CompressedAttention module for testing."""
    rngs = nnx.Rngs(0)
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    attn = attention_compressed.CompressedAttention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.batch_size, self.seq_len, config.emb_dim),
        inputs_kv_shape=(self.batch_size, self.seq_len, config.emb_dim),
        compress_ratio=self.compress_ratio,
        q_lora_rank=config.q_lora_rank,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )
    return attn

  def test_csa_indexer_loss_computation(self):
    """Test that CSA forward pass computes and stores indexer_loss variable."""
    config = self._get_config(indexer_loss_scaling_factor=0.5, indexer_sparse_training=False)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    out, _ = attn(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNotNone(out)
    self.assertEqual(out.shape, (self.batch_size, self.seq_len, config.emb_dim))
    self.assertTrue(hasattr(attn, "indexer_loss"))
    self.assertIsInstance(attn.indexer_loss, indexer_losses)

    loss_val = attn.indexer_loss.get_value()
    self.assertGreater(float(loss_val), 0.0)

  def test_csa_indexer_loss_sparse_training_mode(self):
    """Test CSA forward pass and indexer loss in sparse pre-training mode."""
    config = self._get_config(indexer_loss_scaling_factor=0.5, indexer_sparse_training=True)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(3), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(4), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    out, _ = attn(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNotNone(out)
    self.assertTrue(hasattr(attn, "indexer_loss"))
    self.assertIsInstance(attn.indexer_loss, indexer_losses)
    self.assertGreater(float(attn.indexer_loss.get_value()), 0.0)

  def test_csa_indexer_loss_kl_divergence_zero(self):
    """Test KL divergence is 0 when predicted and target distributions match."""
    config = self._get_config(indexer_loss_scaling_factor=1.0)
    attn = self._init_csa_attention(config)

    n_windows = self.seq_len // self.compress_ratio
    query = jnp.zeros((self.batch_size, self.seq_len, config.num_query_heads, config.head_dim))
    compressed_kv = jnp.zeros((self.batch_size, n_windows, config.num_kv_heads, config.head_dim))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    # Causal block mask (matches what DeepseekV4Indexer produces on uniform logits)
    q_pos = jnp.arange(self.seq_len)[:, None]
    block_end_pos = (jnp.arange(n_windows)[None, :] + 1) * self.compress_ratio
    future_mask = block_end_pos > (q_pos + 1)
    indexer_score = jnp.where(future_mask[None, :, :], DEFAULT_MASK_VALUE, 0.0)

    loss = attn.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-5)

  def test_csa_indexer_loss_head_chunking_parity(self):
    """Test that head chunking scan produces loss mathematically equivalent within FP tolerance to native einsum."""
    config_chunked = self._get_config(mla_qk_head_chunk_size=2)
    config_native = self._get_config(mla_qk_head_chunk_size=0)
    attn_chunked = self._init_csa_attention(config_chunked)
    attn_native = self._init_csa_attention(config_native)

    n_windows = self.seq_len // self.compress_ratio
    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)
    query = jax.random.normal(
        k1, (self.batch_size, self.seq_len, config_chunked.num_query_heads, config_chunked.head_dim)
    )
    compressed_kv = jax.random.normal(
        k2, (self.batch_size, n_windows, config_chunked.num_kv_heads, config_chunked.head_dim)
    )
    indexer_score = jax.random.normal(k3, (self.batch_size, self.seq_len, n_windows))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    loss_chunked = attn_chunked.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    loss_native = attn_native.calculate_csa_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=None,
        position_ids=None,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss_chunked), float(loss_native), rtol=1e-5, atol=1e-5)

  def test_csa_indexer_scoring_head_chunking_parity(self):
    """Test that indexer forward scoring produces identical top-k indices and scores with chunking."""
    config_chunked = self._get_config(mla_qk_head_chunk_size=2)
    config_native = self._get_config(mla_qk_head_chunk_size=0)
    attn_chunked = self._init_csa_attention(config_chunked)
    attn_native = self._init_csa_attention(config_native)

    idx_chunked = attn_chunked.csa_compressor.indexer
    idx_native = attn_native.csa_compressor.indexer
    nnx.update(idx_chunked, nnx.state(idx_native))

    hidden_states = jax.random.normal(jax.random.PRNGKey(10), (self.batch_size, self.seq_len, config_native.emb_dim))
    q_latent = jax.random.normal(jax.random.PRNGKey(11), (self.batch_size, self.seq_len, config_native.q_lora_rank))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))

    topk_chunked, scores_chunked = idx_chunked(
        hidden_states=hidden_states,
        q_latent=q_latent,
        position_ids=positions,
        model_mode=MODEL_MODE_TRAIN,
        return_scores=True,
    )
    topk_native, scores_native = idx_native(
        hidden_states=hidden_states,
        q_latent=q_latent,
        position_ids=positions,
        model_mode=MODEL_MODE_TRAIN,
        return_scores=True,
    )
    np.testing.assert_array_equal(np.array(topk_chunked), np.array(topk_native))
    np.testing.assert_allclose(np.array(scores_chunked), np.array(scores_native), rtol=1e-5, atol=1e-5)

  def test_csa_indexer_chunked_gradients_flow(self):
    """Test that gradients flow through indexer under head chunking with jax.checkpoint rematerialization."""
    config = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=False, mla_qk_head_chunk_size=2)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    def loss_fn(attn_model, q, kv, seg=segment_ids, pos=positions):
      attn_model(
          inputs_q=q,
          inputs_kv=kv,
          decoder_segment_ids=seg,
          inputs_positions=pos,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return attn_model.indexer_loss.get_value()

    grad_model_fn = nnx.grad(loss_fn, argnums=0)
    grads = grad_model_fn(attn, inputs_q, inputs_kv)

    self.assertIsNotNone(grads.csa_compressor.indexer.q_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.kv_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.gate_proj.kernel)
    self.assertIsNotNone(grads.csa_compressor.indexer.weights_proj.kernel)

    q_grad_norm = jnp.linalg.norm(grads.csa_compressor.indexer.q_proj.kernel.get_value())
    self.assertGreater(float(q_grad_norm), 0.0)
    self.assertGreater(float(jnp.linalg.norm(grads.csa_compressor.indexer.weights_proj.kernel.get_value())), 0.0)

    # Gradients must not leak into main model projections
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_a.kernel.get_value())), 0.0)
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_b.kernel.get_value())), 0.0)
    self.assertAlmostEqual(float(jnp.linalg.norm(grads.wkv.kernel.get_value())), 0.0)

    # Gradients with respect to inputs must be zero
    grad_inputs_fn = nnx.grad(loss_fn, argnums=(1, 2))
    grad_q, grad_kv = grad_inputs_fn(attn, inputs_q, inputs_kv)
    self.assertAlmostEqual(float(jnp.linalg.norm(grad_q)), 0.0)
    self.assertAlmostEqual(float(jnp.linalg.norm(grad_kv)), 0.0)

  def test_csa_indexer_gradients_flow(self):
    """Test that gradients flow to indexer parameters and do not leak into main projections or inputs."""
    for is_sparse in (False, True):
      with self.subTest(indexer_sparse_training=is_sparse):
        config = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=is_sparse)
        attn = self._init_csa_attention(config)

        inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
        inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
        positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
        segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

        def loss_fn(attn_model, q, kv, seg=segment_ids, pos=positions):
          attn_model(
              inputs_q=q,
              inputs_kv=kv,
              decoder_segment_ids=seg,
              inputs_positions=pos,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
          )
          return attn_model.indexer_loss.get_value()

        # 1. Gradients with respect to model parameters (argnums=0)
        grad_model_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_model_fn(attn, inputs_q, inputs_kv)

        # Gradients must flow to indexer projection kernels
        self.assertIsNotNone(grads.csa_compressor.indexer.q_proj.kernel)
        self.assertIsNotNone(grads.csa_compressor.indexer.kv_proj.kernel)
        self.assertIsNotNone(grads.csa_compressor.indexer.gate_proj.kernel)
        self.assertIsNotNone(grads.csa_compressor.indexer.weights_proj.kernel)

        q_grad_norm = jnp.linalg.norm(grads.csa_compressor.indexer.q_proj.kernel.get_value())
        self.assertGreater(float(q_grad_norm), 0.0)

        # Gradients must not leak into main model projections
        self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_a.kernel.get_value())), 0.0)
        self.assertAlmostEqual(float(jnp.linalg.norm(grads.wq_b.kernel.get_value())), 0.0)
        self.assertAlmostEqual(float(jnp.linalg.norm(grads.wkv.kernel.get_value())), 0.0)

        # 2. Gradients with respect to inputs (argnums=(1, 2)) must be zero (detached)
        grad_inputs_fn = nnx.grad(loss_fn, argnums=(1, 2))
        grad_q, grad_kv = grad_inputs_fn(attn, inputs_q, inputs_kv)
        self.assertAlmostEqual(float(jnp.linalg.norm(grad_q)), 0.0)
        self.assertAlmostEqual(float(jnp.linalg.norm(grad_kv)), 0.0)

  def test_dense_warmup_forward_mask_is_causal_dense(self):
    """Test that dense warm-up forward pass executes the dense causal path.

    Asserts mask values and compares against top-k=1 sparse mode.
    """
    # Case A: Verify default pre-training (scale=0) executes cleanly without registering indexer loss
    config_unscaled = self._get_config(indexer_loss_scaling_factor=0.0, indexer_sparse_training=False)
    attn_unscaled = self._init_csa_attention(config_unscaled)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config_unscaled.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config_unscaled.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    n_windows = self.seq_len // self.compress_ratio

    attn_unscaled(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    # No indexer loss registered when scaling_factor == 0.0
    self.assertFalse(hasattr(attn_unscaled, "indexer_loss"))
    self.assertIsNone(getattr(attn_unscaled, "indexer_loss", None))

    # Directly assert the dense causal mask values across block boundaries
    dense_mask = attn_unscaled.get_compressed_mask(positions, n_windows)
    self.assertEqual(dense_mask.shape, (self.batch_size, 1, self.seq_len, n_windows))
    # Token t=0: All 4 blocks are future -> all masked
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 0, :]), DEFAULT_MASK_VALUE, atol=1e-5)
    # Token t=3: Block 0 complete (valid 0.0), Blocks 1..3 future (DEFAULT_MASK_VALUE)
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 3, 0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 3, 1:]), DEFAULT_MASK_VALUE, atol=1e-5)
    # Token t=7: Blocks 0..1 complete (valid 0.0), Blocks 2..3 future (DEFAULT_MASK_VALUE)
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 7, :2]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 7, 2:]), DEFAULT_MASK_VALUE, atol=1e-5)
    # Token t=15: All blocks 0..3 complete -> all 0.0
    np.testing.assert_allclose(np.array(dense_mask[:, 0, 15, :]), 0.0, atol=1e-5)

    # Case B: Output divergence between dense warm-up and top-1 sparse mode
    config_dense = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=False, indexer_topk=1)
    config_sparse = self._get_config(indexer_loss_scaling_factor=1.0, indexer_sparse_training=True, indexer_topk=1)

    attn_dense = self._init_csa_attention(config_dense)
    attn_sparse = self._init_csa_attention(config_sparse)

    state_dense = nnx.state(attn_dense)
    nnx.update(attn_sparse, state_dense)

    out_dense, _ = attn_dense(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out_sparse, _ = attn_sparse(
        inputs_q=inputs_q,
        inputs_kv=inputs_kv,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    diff_t15 = jnp.linalg.norm(out_dense[:, 15, :] - out_sparse[:, 15, :])
    self.assertGreater(float(diff_t15), 0.05)

    # Loss must be populated in dense warm-up mode when scaling_factor > 0.0
    self.assertIsNotNone(attn_dense.indexer_loss)
    self.assertGreater(float(attn_dense.indexer_loss.get_value()), 0.0)

  def test_teacher_causality_and_packing_on_loss_function(self):
    """Test calculate_csa_indexer_loss directly on a 2-segment packed sequence with causal boundaries."""
    config = self._get_config(indexer_loss_scaling_factor=1.0)
    attn = self._init_csa_attention(config)

    # 2 segments: Doc 1 = tokens 0..7 (blocks 0, 1), Doc 2 = tokens 8..15 (blocks 2, 3)
    n_windows = self.seq_len // self.compress_ratio  # 4 blocks
    positions = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]] * self.batch_size)
    segment_ids = jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]] * self.batch_size)

    # Build compressed_segment_mask for the 2 documents
    comp_seg_ids = jnp.array([[1, 1, 2, 2]] * self.batch_size)
    valid_comp_seg = segment_ids[:, :, None] == comp_seg_ids[:, None, :]
    compressed_segment_mask = jnp.where(valid_comp_seg, 0.0, DEFAULT_MASK_VALUE)

    query = jnp.zeros((self.batch_size, self.seq_len, config.num_query_heads, config.head_dim))
    compressed_kv = jnp.zeros((self.batch_size, n_windows, config.num_kv_heads, config.head_dim))
    compressed_mask = jnp.zeros((self.batch_size, 1, self.seq_len, n_windows))

    # Ground truth student prediction matching causal + packed teacher distribution
    usable_len = n_windows * attn.compress_ratio
    block_positions = positions[:, : usable_len : attn.compress_ratio]
    is_future = (block_positions[:, None, :] + attn.compress_ratio) > (positions[:, :, None] + 1)
    causal_mask = jnp.where(is_future, DEFAULT_MASK_VALUE, 0.0)
    ground_truth_student_scores = causal_mask + compressed_segment_mask

    loss_perfect = attn.calculate_csa_indexer_loss(
        indexer_score=ground_truth_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    np.testing.assert_allclose(float(loss_perfect), 0.0, atol=1e-5)

    # Case B: Student predicts mass on a future block in Doc 1 (t=4 predicting block 1)
    leaky_student_scores = ground_truth_student_scores.at[:, 4, 1].set(100.0)
    loss_future_leak = attn.calculate_csa_indexer_loss(
        indexer_score=leaky_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    self.assertGreater(float(loss_future_leak), 0.1)

    # Case C: Student in Doc 2 predicts mass on a block from Doc 1 (t=12 predicting block 0)
    cross_doc_student_scores = ground_truth_student_scores.at[:, 12, 0].set(100.0)
    loss_cross_doc = attn.calculate_csa_indexer_loss(
        indexer_score=cross_doc_student_scores,
        query=query,
        compressed_kv=compressed_kv,
        compressed_mask=compressed_mask,
        segment_mask=compressed_segment_mask,
        position_ids=positions,
        sparse_loss=False,
        scaling_factor=1.0,
    )
    self.assertGreater(float(loss_cross_doc), 0.1)

  def test_csa_indexer_loss_jit_compile(self):
    """Compile smoke test: verifies that jitting forward pass with CSA indexer loss executes cleanly."""
    config = self._get_config(indexer_loss_scaling_factor=0.5, indexer_sparse_training=False)
    attn = self._init_csa_attention(config)

    inputs_q = jax.random.normal(jax.random.PRNGKey(1), (self.batch_size, self.seq_len, config.emb_dim))
    inputs_kv = jax.random.normal(jax.random.PRNGKey(2), (self.batch_size, self.seq_len, config.emb_dim))
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    segment_ids = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)

    @nnx.jit
    def jitted_forward(attn_model, q, kv, seg, pos):
      out, _ = attn_model(
          inputs_q=q,
          inputs_kv=kv,
          decoder_segment_ids=seg,
          inputs_positions=pos,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return out, attn_model.indexer_loss.get_value()

    out, loss_val = jitted_forward(attn, inputs_q, inputs_kv, segment_ids, positions)
    self.assertEqual(out.shape, (self.batch_size, self.seq_len, config.emb_dim))
    self.assertGreater(float(loss_val), 0.0)

  def test_mask_routing_matrix(self):
    """Verify mask routing selects sparse mask by default, and dense only during active dense warmup."""
    positions = jnp.broadcast_to(jnp.arange(self.seq_len)[None, :], (self.batch_size, self.seq_len))
    sparse_mask = jnp.full((self.batch_size, 1, self.seq_len, 4), DEFAULT_MASK_VALUE)

    # (mode, scaling_factor, sparse_training, expected_is_sparse)
    test_matrix = [
        (MODEL_MODE_TRAIN, 0.0, False, True),  # Default pre-training: sparse mask
        (MODEL_MODE_TRAIN, 1.0, False, False),  # Active dense warm-up: dense causal mask
        (MODEL_MODE_TRAIN, 1.0, True, True),  # Sparse training: sparse mask
        (MODEL_MODE_PREFILL, 0.0, False, True),  # Prefill inference: sparse mask
        (MODEL_MODE_AUTOREGRESSIVE, 0.0, False, True),  # AR decode inference: sparse mask
    ]

    for mode, scale, sparse_training, expected_sparse in test_matrix:
      config = self._get_config(indexer_loss_scaling_factor=scale, indexer_sparse_training=sparse_training)
      attn = self._init_csa_attention(config)
      is_dense_warmup = (mode == MODEL_MODE_TRAIN) and (scale > 0.0) and (not sparse_training)
      use_sparse_mask = not is_dense_warmup
      routed = attn.get_compressed_mask(positions, 4, sparse_compressed_mask=sparse_mask if use_sparse_mask else None)

      if expected_sparse:
        np.testing.assert_allclose(np.array(routed), np.array(sparse_mask), atol=1e-5)
      else:
        np.testing.assert_allclose(np.array(routed[:, 0, 15, :]), 0.0, atol=1e-5)

  def test_pre_train_loss_fn_stages(self):
    """Verify pre_train.loss_fn behavior across dense, warm-up, and sparse training stages."""
    data = {
        "inputs": jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32),
        "inputs_position": jnp.broadcast_to(jnp.arange(self.seq_len), (self.batch_size, self.seq_len)),
        "inputs_segmentation": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
        "targets": jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
    }
    # Stage 1: Standard dense pre-training (use_indexer=False)
    # Must compute normal LM loss (xent_sum > 0)
    cfg_dense = self._get_config(use_indexer=False, indexer_loss_scaling_factor=0.0, indexer_sparse_training=False)
    mock_model = _MockNnxDecoder(vocab_size=cfg_dense.vocab_size)
    loss_dense, aux_dense = pre_train.loss_fn(mock_model, cfg_dense, data, None, None, is_train=True)
    self.assertGreater(float(aux_dense["xent_sum"]), 0.0)
    self.assertGreater(float(loss_dense), 0.0)

    # Stage 2: Dense warm-up configuration (use_indexer=True, scaling_factor=1.0, sparse_training=False)
    # Must zero out main model LM loss (xent_sum == 0.0)
    cfg_warmup = self._get_config(use_indexer=True, indexer_loss_scaling_factor=1.0, indexer_sparse_training=False)
    _, aux_warmup = pre_train.loss_fn(mock_model, cfg_warmup, data, None, None, is_train=True)
    self.assertEqual(float(aux_warmup["xent_sum"]), 0.0)
    self.assertEqual(float(aux_warmup["z_loss"]), 0.0)

    # Stage 3: Sparse training configuration (use_indexer=True, scaling_factor=1.0, sparse_training=True)
    # Must compute normal LM loss (xent_sum > 0)
    cfg_sparse = self._get_config(use_indexer=True, indexer_loss_scaling_factor=1.0, indexer_sparse_training=True)
    loss_sparse, aux_sparse = pre_train.loss_fn(mock_model, cfg_sparse, data, None, None, is_train=True)
    self.assertGreater(float(aux_sparse["xent_sum"]), 0.0)
    self.assertGreater(float(loss_sparse), 0.0)


if __name__ == "__main__":
  unittest.main()

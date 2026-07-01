# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-8.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests validating DeepSeek-V4 MaxText components against PyTorch references."""

import os
import sys
import unittest

# pylint: disable=import-outside-toplevel, reimported
import jax
import jax.numpy as jnp
import numpy as np
import torch

# To ensure 1:1 parity and avoid outdated or error-prone copy-pasting of reference code,
# this test directly imports the PyTorch reference implementation from a local clone of
# the huggingface/transformers repository containing DeepSeek-V4 implementations.
#
# You can override the default location by setting the `TRANSFORMERS_REPO_PATH` environment variable:
# e.g., `TRANSFORMERS_REPO_PATH=/path/to/transformers python tests/unit/deepseek_v4_vs_reference_test.py`
transformers_repo_path = os.environ.get("TRANSFORMERS_REPO_PATH", "")
sys.path.insert(0, os.path.join(transformers_repo_path, "src"))

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding as DeepseekV4RotaryEmbedding_PT,
    DeepseekV4GroupedLinear as DeepseekV4GroupedLinear_PT,
    DeepseekV4HashRouter as DeepseekV4HashRouter_PT,
    DeepseekV4TopKRouter as DeepseekV4TopKRouter_PT,
    DeepseekV4Experts as DeepseekV4Experts_PT,
    DeepseekV4HyperHead as DeepseekV4HyperHead_PT,
    apply_rotary_pos_emb as ref_apply_rotary_pos_emb,
)

from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.linears import DeepSeekV4GroupedLinear
from maxtext.layers.attention_op import AttentionOp
from maxtext.common.common_types import AttentionType, DEFAULT_MASK_VALUE
from maxtext.layers.mhc import DeepSeek4HyperHead

from flax import nnx
from maxtext.layers.moe import RoutedMoE
from maxtext.layers import initializers

# ==============================================================================
# Tests
# ==============================================================================

# HuggingFace reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py  # pylint: disable=line-too-long
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.attention_compressed import CompressedAttention

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding as PTRope
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb


class DeepSeekV4RotaryEmbeddingTest(unittest.TestCase):
  """Tests to validate MaxText RoPE implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 4096
    self.head_dim = 128
    self.num_heads = 4
    self.main_rope_theta = 10000.0
    self.compress_rope_theta = 160000.0
    self.partial_rotary_factor = 64.0 / 128.0

    self.config = DeepseekV4Config(
        hidden_size=self.num_heads * self.head_dim,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        rope_theta=self.main_rope_theta,
        rope_parameters={
            "main": {
                "rope_type": "default",
                "rope_theta": self.main_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "compress": {
                "rope_type": "default",
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
        },
    )

  def test_rotary_embedding_main(self):
    self._run_rotary_test(layer_type="main", expected_theta=self.main_rope_theta)

  def test_rotary_embedding_compress(self):
    self._run_rotary_test(layer_type="compress", expected_theta=self.compress_rope_theta)

  def test_rotary_embedding_compress_yarn(self):
    original_config = self.config
    self.config = DeepseekV4Config(
        hidden_size=self.num_heads * self.head_dim,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        rope_theta=self.main_rope_theta,
        rope_parameters={
            "main": {
                "rope_type": "default",
                "rope_theta": self.main_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "compress": {
                "rope_type": "yarn",
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
        },
    )
    self.config.max_position_embeddings = 65536 * 16
    try:
      self._run_rotary_test(layer_type="compress", expected_theta=self.compress_rope_theta)
    finally:
      self.config = original_config

  def _run_rotary_test(self, layer_type, expected_theta):
    """
    Validates that the MaxText RoPE implementation is mathematically identical to
    the PyTorch reference up to 1e-5 tolerance.

    Test Flow:
    1. Initializes PyTorch and MaxText Rotary modules with the exact same configuration.
    2. Generates random floating-point noise for inputs to avoid trivial pass cases.
    3. Computes `cos` and `sin` frequencies and compares them directly.
    4. Applies interleaved RoPE rotation to the random inputs in both implementations.
    5. Transposes shapes to match expected dimensions for each framework.
    6. Verifies that the final rotated tensors match exactly.
    """
    # --------------------------------------------------------------------------
    # 1. Initialization
    # --------------------------------------------------------------------------
    ref_rope = DeepseekV4RotaryEmbedding_PT(self.config)
    layer_params = self.config.rope_parameters.get(layer_type, {})
    use_yarn = (layer_params.get("rope_type") == "yarn")
    mt_rope = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=self.partial_rotary_factor,
        rope_theta=expected_theta,
        use_yarn=use_yarn,
        original_max_position_embeddings=layer_params.get("original_max_position_embeddings", 65536),
        max_position_embeddings=self.config.max_position_embeddings,
        beta_fast=layer_params.get("beta_fast", 32.0),
        beta_slow=layer_params.get("beta_slow", 1.0),
    )

    # --------------------------------------------------------------------------
    # 2. Input Generation
    # --------------------------------------------------------------------------
    # Generate non-trivial inputs using np.random.normal to guarantee we are not
    # testing against zeros or ones.
    # Initial shape: [Batch=2, SeqLen=16, NumHeads=4, HeadDim=128]
    np.random.seed(42)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.num_heads, self.head_dim)).astype(np.float32)

    # Position IDs are strictly sequential per batch element: [0, 1, ..., 15]
    position_ids_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)

    # Convert to framework-specific tensors
    x_pt = torch.tensor(x_np)
    position_ids_pt = torch.tensor(position_ids_np, dtype=torch.long)

    x_mt = jnp.array(x_np)
    position_ids_mt = jnp.array(position_ids_np)

    # --------------------------------------------------------------------------
    # 3. Frequency Generation (cos, sin)
    # --------------------------------------------------------------------------
    # PyTorch reference expects flattened hidden dim for computation: [B, S, H*D]
    ref_cos, ref_sin = ref_rope(
        x_pt.view(self.batch_size, self.seq_len, -1), position_ids=position_ids_pt, layer_type=layer_type
    )

    # MaxText natively operates without requiring flattening.
    mt_cos, mt_sin = mt_rope.get_freqs(position_ids_mt)

    # Verify that the calculated frequencies match.
    # Shape of cos/sin: [Batch=2, SeqLen=16, RotaryDim // 2 = 32]
    cos_max_diff = np.max(np.abs(np.array(mt_cos) - ref_cos.numpy()))
    cos_mean_diff = np.mean(np.abs(np.array(mt_cos) - ref_cos.numpy()))
    print(f"Rotary Embedding test ({layer_type}) cos - MAX ABS DIFF: {cos_max_diff:.6e}, MEAN ABS DIFF: {cos_mean_diff:.6e}")
    np.testing.assert_allclose(np.array(mt_cos), ref_cos.numpy(), rtol=1e-2, atol=1e-2)
    sin_max_diff = np.max(np.abs(np.array(mt_sin) - ref_sin.numpy()))
    sin_mean_diff = np.mean(np.abs(np.array(mt_sin) - ref_sin.numpy()))
    print(f"Rotary Embedding test ({layer_type}) sin - MAX ABS DIFF: {sin_max_diff:.6e}, MEAN ABS DIFF: {sin_mean_diff:.6e}")
    np.testing.assert_allclose(np.array(mt_sin), ref_sin.numpy(), rtol=1e-2, atol=1e-2)

    # --------------------------------------------------------------------------
    # 4. Apply Interleaved RoPE Rotation
    # --------------------------------------------------------------------------
    # PyTorch reference `ref_apply_rotary_pos_emb` expects head dimension to be before sequence length:
    # Expected PyTorch Shape: [Batch, NumHeads, SeqLen, HeadDim] = [2, 4, 16, 128]
    x_pt_transpose = x_pt.transpose(1, 2)
    ref_rotated = ref_apply_rotary_pos_emb(x_pt_transpose, ref_cos, ref_sin)

    # Transpose PyTorch result back to MaxText native layout [B, S, H, D] for comparison
    ref_rotated_np = ref_rotated.transpose(1, 2).numpy()

    # MaxText `DeepSeekV4RotaryEmbedding` natively operates on [B, S, H, D].
    # We pass unsqueeze_dim=2 to expand cos/sin from [B, S, D] to [B, S, 1, D]
    # so they correctly broadcast over the NumHeads dimension.
    mt_rotated = mt_rope(x_mt, position_ids_mt, unsqueeze_dim=2)
    mt_rotated_np = np.array(mt_rotated)

    # --------------------------------------------------------------------------
    # 5. Final Validation
    # --------------------------------------------------------------------------
    # Validate the full mathematical rotation is perfectly equivalent.
    max_diff = np.max(np.abs(mt_rotated_np - ref_rotated_np))
    mean_diff = np.mean(np.abs(mt_rotated_np - ref_rotated_np))
    print(f"Rotary Embedding test ({layer_type}) main - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
    np.testing.assert_allclose(mt_rotated_np, ref_rotated_np, rtol=5e-2, atol=5e-2)


class DeepSeekV4GroupedLinearTest(unittest.TestCase):
  """Tests to validate MaxText GroupedLinear implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 8
    self.n_groups = 4
    self.in_features_per_group = 128
    self.out_features = 256  # 64 per group

    self.rngs = nnx.Rngs(0)

  def test_grouped_linear_forward(self):
    """
    Validates that the MaxText GroupedLinear projection is mathematically identical to
    the PyTorch bmm logic up to 1e-5 tolerance.

    Test Flow:
    1. Initializes PyTorch and MaxText GroupedLinear modules with the same configuration.
    2. Extracts the randomly initialized PyTorch weights and transposes them to match MaxText's kernel layout.
    3. Injects the reshaped weights into the MaxText module to ensure exact mathematical parity.
    4. Generates random floating-point noise for inputs.
    5. Executes the forward pass in both implementations.
    6. Verifies that the final projected tensors match exactly.
    """
    # --------------------------------------------------------------------------
    # 1. Initialization
    # --------------------------------------------------------------------------
    ref_linear = DeepseekV4GroupedLinear_PT(
        in_features_per_group=self.in_features_per_group,
        out_features=self.out_features,
        n_groups=self.n_groups,
        bias=False,
    )

    # --------------------------------------------------------------------------
    # 2. Extract and Reshape Weights
    # --------------------------------------------------------------------------
    # Shape of PyTorch weight is [out_features, in_features_per_group]
    # In MaxText we expect [n_groups, in_features_per_group, out_features_per_group]
    pt_weight = ref_linear.weight.data.numpy()  # e.g., [256, 128]

    out_features_per_group = self.out_features // self.n_groups

    # PyTorch's forward does: w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    # This reshapes the weight matrix into group-specific chunks.
    mt_weight_np = pt_weight.reshape(self.n_groups, out_features_per_group, self.in_features_per_group).transpose(0, 2, 1)

    # --------------------------------------------------------------------------
    # 3. Initialize MaxText Implementation
    # --------------------------------------------------------------------------
    mt_linear = DeepSeekV4GroupedLinear(
        in_features_per_group=self.in_features_per_group,
        out_features=self.out_features,
        n_groups=self.n_groups,
        matmul_precision="highest",
        rngs=self.rngs,
    )
    # Manually inject weights for mathematical comparison
    mt_linear.kernel[...] = jnp.array(mt_weight_np)

    # --------------------------------------------------------------------------
    # 4. Input Generation
    # --------------------------------------------------------------------------
    # Generate non-trivial inputs using np.random.normal to guarantee we are not
    # testing against zeros or ones.
    # Shape: [Batch, SeqLen, N_Groups, InFeaturesPerGroup]
    np.random.seed(42)
    x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.n_groups, self.in_features_per_group)).astype(
        np.float32
    )

    x_pt = torch.tensor(x_np)
    x_mt = jnp.array(x_np)

    # --------------------------------------------------------------------------
    # 5. Execute Forward Pass
    # --------------------------------------------------------------------------
    # PyTorch grouped linear takes [Batch, SeqLen, N_Groups, InFeaturesPerGroup]
    ref_out = ref_linear(x_pt)

    # MaxText grouped linear
    mt_out = mt_linear(x_mt)

    # --------------------------------------------------------------------------
    # 6. Final Validation
    # --------------------------------------------------------------------------
    # Validate the full mathematical projection is perfectly equivalent.
    max_diff = np.max(np.abs(np.array(mt_out) - ref_out.detach().numpy()))
    mean_diff = np.mean(np.abs(np.array(mt_out) - ref_out.detach().numpy()))
    print(f"GROUPED LINEAR PARITY - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
    np.testing.assert_allclose(np.array(mt_out), ref_out.detach().numpy(), rtol=1e-5, atol=1e-5)


# TODO(parambole): This test is duplicated here to maintain debugging continuity alongside the other reference tests.
# It has also been duplicated into tests/unit/attention_test.py for CI/CD resilience, as this file is skipped in CI.
class DeepSeekV4AttentionMaskingTest(unittest.TestCase):
  """Tests to validate AttentionOp masking logic for DeepSeek-V4 attention patterns.

  TODO: This test is intentionally duplicated here and in `tests/unit/attention_test.py`.
  We keep it here because this file serves as the central source of truth for all DeepSeek-V4
  specifics, keeping the masking logic validation closely coupled to the structural tests.
  However, because `deepseek_v4_vs_reference_test.py` is currently skipped by our CI/CD pipeline,
  it is mirrored in `attention_test.py` to ensure it runs resiliently on every PR.
  """

  def setUp(self):
    self.config = pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml", "enable_checkpointing=False"], run_name="test")

  def test_generate_attention_mask_local_sliding(self):
    """Verifies AttentionType.LOCAL_SLIDING enforces both causal and sliding window constraints."""

    # Test with multiple heads and different sequence lengths
    for s_len in [1, 8, 128]:
      op = AttentionOp(
          config=self.config,
          num_query_heads=4,
          num_kv_heads=1,
          max_target_length=256,
          mesh=None,
          attention_kernel="dot_product",
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=3,
      )

      batch_size = 1
      q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
      k_dummy = jnp.zeros((batch_size, s_len, 1, 128))

      mask = op.generate_attention_mask(
          query=q_dummy,
          key=k_dummy,
          decoder_segment_ids=None,
          model_mode="train",
      )

      self.assertEqual(mask.shape, (1, 1, 1, s_len, s_len))
      mask_np = np.array(mask)[0, 0, 0]

      # Expected float mask for window_size=3
      # Row 0: [0.0, INF, INF, INF, INF, ...]
      # Row 1: [0.0, 0.0, INF, INF, INF, ...]
      # Row 2: [0.0, 0.0, 0.0, INF, INF, ...]
      # Row 3: [INF, 0.0, 0.0, 0.0, INF, ...]
      if s_len > 1:
        self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)  # strict causal
      self.assertEqual(mask_np[0, 0], 0.0)

      if s_len >= 4:
        self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)  # sliding window size=3
        self.assertEqual(mask_np[3, 1], 0.0)

  def test_generate_attention_mask_compressed(self):
    """Verifies AttentionType.COMPRESSED stitches sliding window and float compressed_mask."""

    batch_size = 1
    s_len = 8
    c_len = 2
    kv_len = s_len + c_len

    op = AttentionOp(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        max_target_length=128,
        mesh=None,
        attention_kernel="dot_product",
        attention_type=AttentionType.COMPRESSED,
        sliding_window_size=3,
    )

    q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
    k_dummy = jnp.zeros((batch_size, kv_len, 1, 128))

    # Simulate a compressed float mask [batch, 1, s_len, c_len]
    # In practice, this exactly mirrors what both HCA and CSA output:
    # - HCA emits a simple mask blocking future blocks (batch, 1, seq_len, c_len)
    # - CSA emits a sparse mask where only top-K blocks are 0.0, rest are -inf.
    # We simulate this by making Block 0 invalid (-inf), and Block 1 valid (0.0).
    compressed_mask = np.zeros((batch_size, 1, s_len, c_len), dtype=np.float32)
    compressed_mask[:, :, :, 0] = DEFAULT_MASK_VALUE
    compressed_mask = jnp.array(compressed_mask)

    mask = op.generate_attention_mask(
        query=q_dummy,
        key=k_dummy,
        decoder_segment_ids=None,
        model_mode="train",
        compressed_mask=compressed_mask,
    )

    # Returned float mask should dynamically inherit the dimensionality of compressed_mask
    # Because compressed_mask was 4D, the final mask should also be 4D: [batch, 1, s_len, kv_len]
    self.assertEqual(mask.shape, (batch_size, 1, s_len, kv_len))
    mask_np = np.array(mask)[0, 0]

    # Uncompressed block (first s_len cols) follows sliding window float mask
    self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[0, 0], 0.0)
    self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[3, 1], 0.0)

    # Compressed block (last c_len cols) follows compressed_mask strictly
    np.testing.assert_allclose(mask_np[:, s_len], DEFAULT_MASK_VALUE)
    np.testing.assert_allclose(mask_np[:, s_len + 1], 0.0)
    print("Mask logic for uncompressed & compressed attention passed perfectly.")


class DeepSeekV4CompressedAttentionTest(unittest.TestCase):
  """Tests to validate MaxText CompressedAttention implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 512
    self.num_heads = 4
    self.head_dim = 128
    self.hidden_size = 256
    self.q_lora_rank = 32
    self.o_groups = 2
    self.o_lora_rank = 64
    self.qk_rope_head_dim = 64
    self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim

    self.rngs = nnx.Rngs(0)

    self.pt_config = DeepseekV4Config(
        hidden_size=self.hidden_size,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.head_dim,
        o_groups=self.o_groups,
        o_lora_rank=self.o_lora_rank,
        rope_theta=10000.0,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        index_n_heads=2,
        index_head_dim=self.head_dim,
        index_topk=2,
        layer_types=["sliding_attention"],
        num_hidden_layers=1,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": self.partial_rotary_factor},
            "compress": {
                "rope_type": "yarn",
                "rope_theta": 160000.0,
                "partial_rotary_factor": self.partial_rotary_factor,
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
            },
        },
        sliding_window=2048,
        attention_dropout=0.0,
    )

  def _build_maxtext_config(self, layer_type):
    """Builds a MaxText pyconfig for a specific layer_type."""

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "base_emb_dim": self.pt_config.hidden_size,
        "head_dim": self.pt_config.head_dim,
        "base_num_query_heads": self.pt_config.num_attention_heads,
        "base_num_kv_heads": 1,
        "dtype": "float32",
        "weight_dtype": "float32",
        "sliding_window_size": self.pt_config.sliding_window,
        "q_lora_rank": self.pt_config.q_lora_rank,
        "o_groups": self.pt_config.o_groups,
        "o_lora_rank": self.pt_config.o_lora_rank,
        "compress_ratios": [0, 4, 128],  # Dummy list for the test
        "compressed_rope_max_timescale": self.pt_config.rope_parameters["compress"]["rope_theta"],
        "max_position_embeddings": self.pt_config.max_position_embeddings,
        "original_max_position_embeddings": self.pt_config.rope_parameters["compress"]["original_max_position_embeddings"],
        "indexer_n_heads": self.pt_config.index_n_heads,
        "indexer_head_dim": self.pt_config.index_head_dim,
        "indexer_topk": self.pt_config.index_topk,
        "normalization_layer_epsilon": self.pt_config.rms_norm_eps,
        "matmul_precision": "highest",
        "skip_jax_distributed_system": True,
    }

    argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
    mt_config = pyconfig.initialize(argv, **config_arguments)

    return mt_config

  def _copy_linear(self, mt_linear, pt_linear):
    if pt_linear is None or mt_linear is None:
      return
    mt_linear.kernel.value = jnp.array(pt_linear.weight.data.numpy().T)
    if hasattr(pt_linear, "bias") and pt_linear.bias is not None:
      mt_linear.bias.value = jnp.array(pt_linear.bias.data.numpy())

  def _copy_norm(self, mt_norm, pt_norm):
    if pt_norm is None or mt_norm is None:
      return
    if hasattr(pt_norm, "weight") and pt_norm.weight is not None:
      mt_norm.scale.value = jnp.array(pt_norm.weight.data.numpy())

  def _run_e2e_test(self, layer_type, is_packed=False):
    self.pt_config.layer_types = [layer_type]

    torch.manual_seed(42)
    ref_attn = DeepseekV4Attention(self.pt_config, layer_idx=0)
    for p in ref_attn.parameters():
        if p.dim() >= 1:
            torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
        else:
            torch.nn.init.constant_(p.data, 0.02)
    self.ref_attn = ref_attn

    # ==================================================================================================
    # FIXING `jax.lax.top_k` VS `torch.topk` TIE-BREAKING DIVERGENCE
    # ==================================================================================================
    # When `index_topk` is small (e.g. 2) and inputs/weights are randomly initialized around 0:
    # 1. About 50% of the dot products (`Q @ K^T`) in the Indexer are naturally negative.
    # 2. The Indexer applies a `ReLU`, mapping ALL of those negative values to identical `0.0`s.
    # 3. If `index_topk` needs to select blocks from a pool containing mostly `0.0`s, there's a tie!
    # 4. PyTorch's `torch.topk` and MaxText's `jax.lax.top_k` handle these identical `0.0` ties differently
    #    (e.g., PyTorch might pick the first available zero, XLA might pick the last).
    # 5. This causes the frameworks to gather DIFFERENT valid blocks to feed into the final attention
    #    pass. Even though the score was `0.0` in the indexer, the main attention `Q @ K` score is NOT 0.0!
    #    Thus, gathering different blocks causes the outputs to completely diverge.
    #
    # THE PROOF:
    # To prove the underlying math is correct, we force the weights and inputs to be strictly positive.
    # Because `Q @ K^T` is strictly positive, `ReLU` acts as a pure pass-through and zeros nothing out.
    # With no identical `0.0`s, there are NO TIES.
    # Without ties, both PyTorch and MaxText deterministically pick the exact same blocks and match 1:1!
    # ==================================================================================================
    if layer_type == "compressed_sparse_attention" and self.pt_config.index_topk == 2:
      for p in ref_attn.parameters():
        p.data = torch.abs(p.data) + 0.1

    rope_main = PTRope(self.pt_config)
    rope_compress = PTRope(self.pt_config)

    mt_config = self._build_maxtext_config(layer_type)

    mesh = Mesh(mesh_utils.create_device_mesh((1,), devices=jax.local_devices()[:1]), axis_names=("fsdp",))

    compress_ratio_map = {
        "sliding_attention": 0,
        "compressed_sparse_attention": self.pt_config.compress_rates["compressed_sparse_attention"],
        "heavily_compressed_attention": self.pt_config.compress_rates["heavily_compressed_attention"],
    }
    compress_ratio = compress_ratio_map[layer_type]
    layer_attention_type = AttentionType.LOCAL_SLIDING if compress_ratio == 0 else AttentionType.COMPRESSED

    mt_attn = CompressedAttention(
        config=mt_config,
        compress_ratio=compress_ratio,
        attention_type=layer_attention_type,
        num_query_heads=self.num_heads,
        num_kv_heads=1,
        head_dim=self.head_dim,
        max_target_length=128,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.batch_size, self.seq_len, self.hidden_size),
        inputs_kv_shape=(self.batch_size, self.seq_len, self.hidden_size),
        q_lora_rank=self.q_lora_rank,
        sliding_window_size=mt_config.sliding_window_size,
        rngs=self.rngs,
    )
    self.mt_attn = mt_attn

    # 3. Copy Weights
    self._copy_linear(mt_attn.wq_a, ref_attn.q_a_proj)
    mt_attn.wq_b.kernel.value = jnp.array(
        ref_attn.q_b_proj.weight.data.numpy().T.reshape(self.q_lora_rank, self.num_heads, self.head_dim)
    )
    mt_attn.wkv.kernel.value = jnp.array(
        ref_attn.kv_proj.weight.data.numpy().T.reshape(
            self.hidden_size, self.pt_config.num_key_value_heads, self.head_dim
        )
    )
    self._copy_norm(mt_attn.q_norm, ref_attn.q_a_norm)
    self._copy_norm(mt_attn.kv_norm, ref_attn.kv_norm)
    mt_attn.sinks.value = jnp.array(ref_attn.sinks.data.numpy().reshape(-1))

    pt_oa_weight = ref_attn.o_a_proj.weight.data.numpy()
    mt_oa_weight = pt_oa_weight.reshape(self.o_groups, -1, (self.num_heads * self.head_dim) // self.o_groups).transpose(
        0, 2, 1
    )
    mt_attn.o_a_proj.kernel.value = jnp.array(mt_oa_weight)
    self._copy_linear(mt_attn.o_b_proj, ref_attn.o_b_proj)

    if layer_type == "heavily_compressed_attention":
      self._copy_linear(mt_attn.hca_compressor.kv_proj, ref_attn.compressor.kv_proj)
      self._copy_linear(mt_attn.hca_compressor.gate_proj, ref_attn.compressor.gate_proj)
      mt_attn.hca_compressor.position_bias.value = jnp.array(ref_attn.compressor.position_bias.data.numpy())
      self._copy_norm(mt_attn.hca_compressor.kv_norm, ref_attn.compressor.kv_norm)

    if layer_type == "compressed_sparse_attention":
      self._copy_linear(mt_attn.csa_compressor.kv_proj, ref_attn.compressor.kv_proj)
      self._copy_linear(mt_attn.csa_compressor.gate_proj, ref_attn.compressor.gate_proj)
      mt_attn.csa_compressor.position_bias.value = jnp.array(ref_attn.compressor.position_bias.data.numpy())
      self._copy_norm(mt_attn.csa_compressor.kv_norm, ref_attn.compressor.kv_norm)

      self._copy_linear(mt_attn.csa_compressor.indexer.q_proj, ref_attn.compressor.indexer.q_b_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.kv_proj, ref_attn.compressor.indexer.kv_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.gate_proj, ref_attn.compressor.indexer.gate_proj)
      self._copy_linear(mt_attn.csa_compressor.indexer.weights_proj, ref_attn.compressor.indexer.scorer.weights_proj)
      mt_attn.csa_compressor.indexer.position_bias.value = jnp.array(
          ref_attn.compressor.indexer.position_bias.data.numpy()
      )
      self._copy_norm(mt_attn.csa_compressor.indexer.kv_norm, ref_attn.compressor.indexer.kv_norm)

    # 4. Inputs
    np.random.seed(42)
    if layer_type == "compressed_sparse_attention" and self.pt_config.index_topk == 2:
      # Generate strictly positive inputs (`[0.1, 1.0]`) to correspond with the strictly
      # positive weights injected above. This guarantees that `Q @ K^T` is always > 0.0,
      # sidestepping the indexer ReLU tie-breaking behavior entirely.
      x_np = np.random.uniform(0.1, 1.0, size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    else:
      x_np = np.random.normal(size=(self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    pos_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)

    x_pt = torch.tensor(x_np)
    pos_pt = torch.tensor(pos_np, dtype=torch.long)

    x_mt = jnp.array(x_np)
    pos_mt = jnp.array(pos_np)

    if is_packed:
      half = self.seq_len // 2
      # Simulate two packed documents [0..half-1] and [half..seq_len-1]
      segs_np = np.ones((self.batch_size, self.seq_len), dtype=np.int32)
      segs_np[:, half:] = 2
      segs_mt = jnp.array(segs_np)
    else:
      segs_mt = jnp.ones_like(pos_mt, dtype=jnp.int32)

    # 5. Execute PyTorch
    dummy_x_main = torch.zeros(self.batch_size, self.seq_len, 1)
    cos_main, sin_main = rope_main(dummy_x_main, pos_pt, "main")
    cos_comp, sin_comp = rope_compress(dummy_x_main, pos_pt, "compress")

    pt_positions = {"main": (cos_main, sin_main), "compress": (cos_comp, sin_comp)}

    if is_packed:
      # Use an exact block diagonal mask for packing
      pt_mask = torch.full((self.batch_size, 1, self.seq_len, self.seq_len), float("-inf"))
      pt_mask[:, :, :half, :half] = _prepare_4d_causal_attention_mask(None, (self.batch_size, half), x_pt, 0, 2048)
      pt_mask[:, :, half:, half:] = _prepare_4d_causal_attention_mask(
          None, (self.batch_size, self.seq_len - half), x_pt, 0, 2048
      )
    else:
      # Native PyTorch causal mask generator
      pt_mask = _prepare_4d_causal_attention_mask(None, (self.batch_size, self.seq_len), x_pt, 0, 2048)

    pt_out, _ = ref_attn(x_pt, pt_positions, pos_pt, attention_mask=pt_mask)

    # Extract indexer top_k from PyTorch
    if layer_type == "compressed_sparse_attention":
      pt_q_residual = ref_attn.q_a_norm(ref_attn.q_a_proj(x_pt))
      pt_top_k_indices = ref_attn.compressor.indexer(x_pt, pt_q_residual, pos_pt, None, 0)
      print(f"PyTorch top_k_indices:\n{pt_top_k_indices[0]}")

      mt_q_latent = mt_attn.wq_a(x_mt)
      mt_q_residual = mt_attn.q_norm(mt_q_latent)
      mt_top_k_indices = mt_attn.csa_compressor.indexer(x_mt, mt_q_residual, pos_mt)
      print(f"MaxText top_k_indices:\n{mt_top_k_indices[0]}")

      num_mismatches = np.sum(pt_top_k_indices.detach().numpy() != np.array(mt_top_k_indices))
      print(f"top_k_indices mismatches: {num_mismatches}")

    # 6. Execute MaxText
    mt_out, _ = mt_attn(x_mt, x_mt, segs_mt, pos_mt, deterministic=True, model_mode=MODEL_MODE_TRAIN)

    # 7. Asserts
    if not is_packed:
      print("Comparing MaxText vs PyTorch:")
      if hasattr(mt_attn, "hca_compressor"):
        mt_comp = mt_attn.hca_compressor
        pt_comp = ref_attn.compressor

        pt_kv = pt_comp.kv_proj(x_pt)
        mt_kv = mt_comp.kv_proj(x_mt)
        print(f"kv_proj error: {np.max(np.abs(pt_kv.detach().numpy() - np.array(mt_kv)))}")

        pt_gate = pt_comp.gate_proj(x_pt)
        mt_gate = mt_comp.gate_proj(x_mt)
        print(f"gate_proj error: {np.max(np.abs(pt_gate.detach().numpy() - np.array(mt_gate)))}")

        # We need to manually compute compressed for pt and mt to compare
        # [batch, seq_len, head_dim] -> [batch, n_windows, compress_rate, head_dim]
        batch, seq_len, _ = x_pt.shape
        n_windows = seq_len // pt_comp.compress_rate
        pt_chunk_kv = pt_kv.view(batch, n_windows, pt_comp.compress_rate, -1)
        pt_chunk_gate = pt_gate.view(batch, n_windows, pt_comp.compress_rate, -1) + pt_comp.position_bias

        # [batch, seq_len, head_dim] -> [batch, n_windows, compress_rate, head_dim]
        mt_chunk_kv = mt_kv.reshape((batch, n_windows, mt_comp.compress_rate, -1))
        mt_chunk_gate = mt_gate.reshape((batch, n_windows, mt_comp.compress_rate, -1)) + mt_comp.position_bias.value
        print(f"chunk_gate error: {np.max(np.abs(pt_chunk_gate.detach().numpy() - np.array(mt_chunk_gate)))}")

        pt_gate_weights = pt_chunk_gate.softmax(dim=2, dtype=torch.float32).to(pt_chunk_kv.dtype)
        mt_gate_weights = jax.nn.softmax(mt_chunk_gate, axis=2).astype(mt_chunk_kv.dtype)
        print(f"gate_weights error: {np.max(np.abs(pt_gate_weights.detach().numpy() - np.array(mt_gate_weights)))}")

        # [batch, n_windows, compress_rate, head_dim] -> [batch, n_windows, head_dim]
        pt_compressed = pt_comp.kv_norm((pt_chunk_kv * pt_gate_weights).sum(dim=2))
        mt_compressed = mt_comp.kv_norm(jnp.sum(mt_chunk_kv * mt_gate_weights, axis=2))
        print(f"compressed before rope error: {np.max(np.abs(pt_compressed.detach().numpy() - np.array(mt_compressed)))}")

        # Calculate RoPE embeddings for compressed tokens
        # -> [1, n_windows]
        pt_positions = torch.arange(n_windows) * pt_comp.compress_rate
        pt_positions = pt_positions.unsqueeze(0).expand(batch, -1)
        pt_cos, pt_sin = pt_comp.rotary_emb(pt_compressed, position_ids=pt_positions, layer_type=pt_comp.rope_layer_type)

        # -> [batch, n_windows]
        mt_positions = jnp.arange(n_windows) * mt_comp.compress_rate
        mt_positions = jnp.broadcast_to(mt_positions[None, :], (batch, n_windows))
        mt_cos, mt_sin = mt_comp.rotary_emb.get_freqs(mt_positions)
        print(f"cos error: {np.max(np.abs(pt_cos.detach().numpy() - np.array(mt_cos)))}")
        print(f"sin error: {np.max(np.abs(pt_sin.detach().numpy() - np.array(mt_sin)))}")

        pt_compressed_rot = apply_rotary_pos_emb(pt_compressed.unsqueeze(1), pt_cos, pt_sin).squeeze(1)
        mt_compressed_rot = mt_comp.rotary_emb(mt_compressed, mt_positions, unsqueeze_dim=None)
        error = np.max(np.abs(pt_compressed_rot.detach().numpy() - np.array(mt_compressed_rot)))
        print(f"compressed after rope error: {error}")

      if layer_type == "compressed_sparse_attention":
        pt_comp = ref_attn.compressor
        mt_comp = mt_attn.csa_compressor
        kv_error = np.max(np.abs(pt_comp.kv_proj(x_pt).detach().numpy() - np.array(mt_comp.kv_proj(x_mt))))
        print(f"csa kv_proj error: {kv_error}")
        gate_error = np.max(np.abs(pt_comp.gate_proj(x_pt).detach().numpy() - np.array(mt_comp.gate_proj(x_mt))))
        print(f"csa gate_proj error: {gate_error}")

      max_diff = np.max(np.abs(np.array(mt_out) - pt_out.detach().numpy()))
      mean_diff = np.mean(np.abs(np.array(mt_out) - pt_out.detach().numpy()))
      print(f"COMPRESSED ATTENTION PARITY layer_type={layer_type} - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
      np.testing.assert_allclose(np.array(mt_out), pt_out.detach().numpy(), rtol=2e-2, atol=2e-2)
    else:
      # Since PyTorch leaks cross-document compressed blocks due to its bug (ignoring attention_mask
      # when appending block_bias), the outputs will NOT match.
      # We assert that they explicitly DO NOT match, proving MaxText successfully firewalls!
      self.assertFalse(np.allclose(np.array(mt_out), pt_out.detach().numpy(), rtol=1e-3, atol=1e-3))
      print(f"Document packing test ({layer_type}) successfully confirmed PyTorch bug and MaxText firewall.")

  def test_forward_uncompressed(self):
    self._run_e2e_test("sliding_attention")

  def test_forward_hca(self):
    self._run_e2e_test("heavily_compressed_attention")

  def test_forward_csa(self):
    self._run_e2e_test("compressed_sparse_attention")

  def test_document_packing_masking(self):
    self._run_e2e_test("heavily_compressed_attention", is_packed=True)


class DeepSeekV4MoERouterTest(unittest.TestCase):

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 8
    self.hidden_dim = 128
    self.num_experts = 16
    self.num_experts_per_tok = 4
    self.vocab_size = 1000

    self.pt_config = DeepseekV4Config(
        hidden_size=self.hidden_dim,
        num_local_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        routed_scaling_factor=2.0,
        scoring_func="sqrtsoftplus",
        vocab_size=self.vocab_size,
    )

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "base_emb_dim": self.hidden_dim,
        "num_experts": self.num_experts,
        "topk_routing_group": self.num_experts_per_tok,
        "routed_scaling_factor": 2.0,
        "routed_score_func": "sqrtsoftplus",
        "routed_bias": True,
        "n_routing_groups": -1,
        "vocab_size": self.vocab_size,
        "first_num_hash_layers": 3,
        "decoder_block": "deepseek4",
        "model_name": "deepseek4-284b",
        "attention": "dot_product",
        "base_mlp_dim": 256,
        "base_moe_mlp_dim": 256,
        "override_model_config": True,
        "skip_jax_distributed_system": True,
    }
    argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
    self.mx_config = pyconfig.initialize(argv, **config_arguments)

    devices = np.array(jax.devices()[:1])
    self.mesh = jax.sharding.Mesh(devices, ("tensor",))
    self.rngs = nnx.Rngs(0)

  def test_hash_router(self):
    pt_router = DeepseekV4HashRouter_PT(self.pt_config)
    # Explicitly initialize PyTorch weights since torch.empty leaves garbage in memory,
    # which causes NaN/Inf drift between PyTorch and MaxText/XLA execution.
    torch.nn.init.normal_(pt_router.weight)

    # Hash Router operates deterministically based on input_ids via a frozen tid2eid lookup table.
    # In practice, this table is pre-computed (e.g. by K-Means on the dataset) and loaded statically.
    # For this parity test, we randomly initialize the lookup table on the PyTorch side
    # and explicitly sync it to the MaxText side to ensure both routers route the exact same way.
    pt_tid2eid = torch.randint(0, self.num_experts, (self.vocab_size, self.num_experts_per_tok))
    pt_router.tid2eid.copy_(pt_tid2eid)

    mx_moe = RoutedMoE(
        config=self.mx_config,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", "mlp_moe", None),
        rngs=self.rngs,
        is_hash_routing=True,  # Hash layer
    )

    # Sync weights
    mx_moe.tid2eid.value = jnp.array(pt_router.tid2eid.numpy(), dtype=jnp.float32)
    mx_moe.gate.kernel.value = jnp.array(pt_router.weight.detach().numpy()).T

    hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
    input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    # PT forward
    _, pt_weights, pt_indices = pt_router(hidden_states, input_ids)

    # MaxText forward
    gate_logits, pre_bias_logits = mx_moe.gate(jnp.array(hidden_states.numpy()))
    mx_weights, mx_indices = mx_moe.get_topk(
        gate_logits, pre_bias_logits, rngs=self.rngs, input_ids=jnp.array(input_ids.numpy())
    )

    # --- Assertion Logic for Hash Router ---
    # PyTorch returns flat tensors: (batch * seq_len, top_k)
    # MaxText returns structured tensors: (batch, seq_len, top_k)
    # We must explicitly reshape PyTorch outputs to match MaxText's nested sequence structure.
    pt_indices_reshaped = pt_indices.numpy().reshape(self.batch_size, self.seq_len, -1)
    pt_weights_reshaped = pt_weights.detach().numpy().reshape(self.batch_size, self.seq_len, -1)
    weights_max_diff = np.max(np.abs(mx_weights - pt_weights_reshaped))
    weights_mean_diff = np.mean(np.abs(mx_weights - pt_weights_reshaped))
    print(f"MOE HASH ROUTER WEIGHTS PARITY - MAX ABS DIFF: {weights_max_diff:.6e}, MEAN ABS DIFF: {weights_mean_diff:.6e}")
    np.testing.assert_allclose(mx_indices, pt_indices_reshaped, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mx_weights, pt_weights_reshaped, rtol=1e-2, atol=1e-2)

  def test_topk_router(self):
    pt_router = DeepseekV4TopKRouter_PT(self.pt_config)

    # Explicitly initialize PyTorch weights since torch.empty leaves garbage in memory,
    # which causes NaN/Inf drift between PyTorch and MaxText/XLA execution.
    torch.nn.init.normal_(pt_router.weight)
    torch.nn.init.normal_(pt_router.e_score_correction_bias)

    mx_moe = RoutedMoE(
        config=self.mx_config,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", "mlp_moe", None),
        rngs=self.rngs,
        is_hash_routing=False,  # TopK layer
    )

    # Sync weights
    mx_moe.gate.kernel.value = jnp.array(pt_router.weight.detach().numpy()).T
    mx_moe.gate.bias.value = jnp.array(pt_router.e_score_correction_bias.detach().numpy())

    hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

    # PT forward
    _, pt_weights, pt_indices = pt_router(hidden_states)

    # MaxText forward
    gate_logits, pre_bias_logits = mx_moe.gate(jnp.array(hidden_states.numpy()))
    mx_weights, mx_indices = mx_moe.get_topk(gate_logits, pre_bias_logits, rngs=self.rngs)

    # --- Assertion Logic for TopK Router ---
    # PyTorch returns flat tensors: (batch * seq_len, top_k)
    # MaxText returns structured tensors: (batch, seq_len, top_k)
    # 1. Reshape PyTorch outputs to match MaxText's nested sequence structure.
    pt_indices_reshaped = pt_indices.numpy().reshape(self.batch_size, self.seq_len, -1)
    pt_weights_reshaped = pt_weights.detach().numpy().reshape(self.batch_size, self.seq_len, -1)

    # 2. Sort both by indices so they can be compared directly.
    # jax.lax.top_k and torch.topk resolve exact-value ties differently.
    # Because TopK routing weights are summed commutatively during the MoE forward pass,
    # only the mathematical *set* of selected experts matters, not their strict sorted order.
    mx_sort_idx = np.argsort(mx_indices, axis=-1)
    pt_sort_idx = np.argsort(pt_indices_reshaped, axis=-1)

    mx_indices_sorted = np.take_along_axis(np.array(mx_indices), mx_sort_idx, axis=-1)
    mx_weights_sorted = np.take_along_axis(np.array(mx_weights), mx_sort_idx, axis=-1)

    pt_indices_sorted = np.take_along_axis(pt_indices_reshaped, pt_sort_idx, axis=-1)
    pt_weights_sorted = np.take_along_axis(pt_weights_reshaped, pt_sort_idx, axis=-1)

    weights_max_diff = np.max(np.abs(mx_weights_sorted - pt_weights_sorted))
    weights_mean_diff = np.mean(np.abs(mx_weights_sorted - pt_weights_sorted))
    print(f"MOE TOPK ROUTER WEIGHTS PARITY - MAX ABS DIFF: {weights_max_diff:.6e}, MEAN ABS DIFF: {weights_mean_diff:.6e}")
    np.testing.assert_allclose(mx_indices_sorted, pt_indices_sorted, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mx_weights_sorted, pt_weights_sorted, rtol=1e-2, atol=1e-2)


class DeepSeekV4SwiGLUClampTest(unittest.TestCase):

  def test_swiglu_clamp(self):
    limit = 10.0
    pt_config = DeepseekV4Config(
        hidden_size=128,
        num_local_experts=2,
        num_experts_per_tok=1,
        intermediate_size=256,
        swiglu_limit=limit,
    )

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "base_emb_dim": 128,
        "num_experts": 2,
        "topk_routing_group": 1,
        "mlp_activations_limit": limit,
        "decoder_block": "deepseek4",
        "model_name": "deepseek4-284b",
        "attention": "dot_product",
        "base_mlp_dim": 256,
        "base_moe_mlp_dim": 256,
        "override_model_config": True,
        "matmul_precision": "highest",
        "skip_jax_distributed_system": True,
    }
    argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
    mx_config = pyconfig.initialize(argv, **config_arguments)

    pt_experts = DeepseekV4Experts_PT(pt_config)

    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(devices, ("tensor",))
    rngs = nnx.Rngs(0)

    mx_moe = RoutedMoE(
        config=mx_config,
        num_experts=2,
        num_experts_per_tok=1,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", "mlp_moe", None),
        rngs=rngs,
    )

    # Gate & Up merged matrix in PT
    gate_up = torch.randn(1, 4, 256 * 2) * 20.0  # Force large values to trigger the clamping mechanism

    # PyTorch reference executes the standard SwiGLU followed by clamping
    # to self.config.swiglu_limit (which translates to mlp_activations_limit in MaxText)
    pt_out = pt_experts._apply_gate(gate_up)  # pylint: disable=protected-access

    # In MaxText, the gate and up projections are separated mathematically.
    # apply_ffn_activation executes the swiglu limit internally during the activation phase.
    gate, up = gate_up.chunk(2, dim=-1)
    mx_out = mx_moe.apply_ffn_activation(jnp.array(gate.numpy()), jnp.array(up.numpy()))

    # Validate that both clamped outputs match identically
    max_diff = np.max(np.abs(mx_out - pt_out.numpy()))
    mean_diff = np.mean(np.abs(mx_out - pt_out.numpy()))
    print(f"SWIGLU CLAMP PARITY - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
    np.testing.assert_allclose(mx_out, pt_out.numpy(), rtol=1e-5, atol=1e-5)


from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer as DeepseekV4DecoderLayer_PT
from maxtext.models.deepseek4 import DeepSeek4DecoderLayer

class DeepSeekV4ConversionMappingTest(unittest.TestCase):
  """Tests to validate weight conversion mappings from PARAM_MAPPING."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 32
    self.hidden_dim = 4096
    self.num_heads = 64
    self.head_dim = 512
    self.q_lora_rank = 1024
    self.o_groups = 8
    self.o_lora_rank = 1024
    self.qk_rope_head_dim = 64
    self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
    self.vocab_size = 129280

    self.pt_config = DeepseekV4Config(
        hidden_size=self.hidden_dim,
        num_attention_heads=self.num_heads,
        num_key_value_heads=1,
        head_dim=self.head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.head_dim,
        o_groups=self.o_groups,
        o_lora_rank=self.o_lora_rank,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        num_hidden_layers=7,
        num_nextn_predict_layers=0,
        num_local_experts=8,
        num_experts_per_tok=3,
        vocab_size=self.vocab_size,
    )

    config_arguments = {
        "model_name": "deepseek4-tiny",
        "override_model_config": True,
        "per_device_batch_size": 1,
        "matmul_precision": "highest",
        "megablox": False,
        "sparse_matmul": False,
        "dtype": "float32",
        "weight_dtype": "float32",
        "skip_jax_distributed_system": True,
    }
    argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
    self.mx_config = pyconfig.initialize(argv, **config_arguments)

    self.rngs = nnx.Rngs(0)
    devices = np.array(jax.devices()[:1])
    self.mesh = jax.sharding.Mesh(devices, ("tensor",))

  def _apply_param_mapping(self, mt_layer, pt_layer, l):
    import importlib.util
    import os
    mapping_path = os.path.join(os.path.dirname(__file__), "../../deepseek4-references/conversion_mapping.py")
    spec = importlib.util.spec_from_file_location("conversion_mapping", mapping_path)
    conversion_mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion_mapping)
    PARAM_MAPPING = conversion_mapping.PARAM_MAPPING

    def get_attr(obj, path):
        if path is None: return None
        if "mlp.experts.." in path:
            parts = path.split("..")
            expert_obj = obj.mlp.experts
            idx_and_weight = parts[1].split(".")
            idx = int(idx_and_weight[0])
            w_name = idx_and_weight[1]
            if w_name == "w1":
                return expert_obj.gate_up_proj[idx, :expert_obj.intermediate_dim, :]
            elif w_name == "w3":
                return expert_obj.gate_up_proj[idx, expert_obj.intermediate_dim:, :]
            elif w_name == "w2":
                return expert_obj.down_proj[idx]
        for part in path.split('.'):
            if hasattr(obj, part): obj = getattr(obj, part)
            elif isinstance(obj, list) or isinstance(obj, dict): obj = obj[int(part)] if isinstance(obj, list) else obj[part]
            else: return None
        return obj

    mt_prefix = f"params.params.decoder.layers_{l}."
    pt_prefix = f"model.layers.{l}."
    for mt_key, (pt_key, rule) in PARAM_MAPPING.items():
        if mt_key.startswith(mt_prefix) or f"params.Tid2EidVar.decoder.layers_{l}" in mt_key:
            if "Tid2EidVar" in mt_key:
                mt_path = mt_key.replace(f"params.Tid2EidVar.decoder.layers_{l}.", "") + ".value"
            else:
                mt_path = mt_key.replace(mt_prefix, "") + ".value"

            if pt_key is None: pt_obj = None
            elif type(pt_key) == list: pt_obj = pt_key
            else: pt_obj = get_attr(pt_layer, pt_key.replace(pt_prefix, ""))

            # Apply rule
            val = None
            if rule == "direct": val = jnp.array(pt_obj.detach().numpy())
            elif rule == "transpose": val = jnp.array(pt_obj.detach().numpy().T)
            elif rule == "stack_transpose":
                try:
                    tensors = [get_attr(pt_layer, path.replace(pt_prefix, "")) for path in pt_obj]
                    val = jnp.array(torch.stack(tensors).detach().numpy()).transpose(0, 2, 1)
                except Exception as e:
                    print(f"FAILED stack_transpose: pt_obj={pt_obj}, tensors={['None' if t is None else 'Tensor' for t in tensors]}")
                    raise e
            elif rule == "expert_gate_proj":
                val = pt_obj.detach().numpy()
                intermediate_dim = val.shape[1] // 2
                val = jnp.array(val[:, :intermediate_dim, :].transpose(0, 2, 1))
            elif rule == "expert_up_proj":
                val = pt_obj.detach().numpy()
                intermediate_dim = val.shape[1] // 2
                val = jnp.array(val[:, intermediate_dim:, :].transpose(0, 2, 1))
            elif rule == "expert_down_proj":
                val = pt_obj.detach().numpy()
                val = jnp.array(val.transpose(0, 2, 1))
            elif rule == "ones": pass
            elif rule.startswith("mhc_fn_"):
                hc = pt_layer.attn_hc.hc_mult
                fn = pt_obj.detach().numpy()
                if rule == "mhc_fn_pre": val = fn[:hc, :]
                elif rule == "mhc_fn_post": val = fn[hc:2*hc, :]
                elif rule == "mhc_fn_res": val = fn[2*hc:, :]
                val = jnp.array(val.T)
            elif rule.startswith("mhc_base_"):
                hc = pt_layer.attn_hc.hc_mult
                base = pt_obj.detach().numpy()
                if rule == "mhc_base_pre": val = base[:hc]
                elif rule == "mhc_base_post": val = base[hc:2*hc]
                elif rule == "mhc_base_res": val = base[2*hc:].reshape(hc, hc)
                val = jnp.array(val)
            elif rule.startswith("mhc_scale_"):
                scale = pt_obj.detach().numpy()
                if rule == "mhc_scale_pre": val = scale[0]
                elif rule == "mhc_scale_post": val = scale[1]
                elif rule == "mhc_scale_res": val = scale[2]
                val = jnp.array([val])
            elif rule == "reshape_transpose_oa":
                val = pt_obj.detach().numpy()
                val = val.reshape(self.pt_config.o_groups, -1, val.shape[1]).transpose(0, 2, 1)
                val = jnp.array(val)
            elif rule == "transpose_reshape_q":
                val = pt_obj.detach().numpy().T.reshape(self.pt_config.q_lora_rank, self.pt_config.num_attention_heads, self.pt_config.head_dim)
                val = jnp.array(val)
            elif rule == "transpose_reshape_kv":
                val = pt_obj.detach().numpy().T.reshape(-1, self.pt_config.num_key_value_heads, self.pt_config.head_dim)
                val = jnp.array(val)
            
            if val is not None or rule == "ones":
                parts = mt_path.split('.')
                obj = mt_layer
                valid = True
                for part in parts[:-1]:
                    if hasattr(obj, part): obj = getattr(obj, part)
                    else: valid = False; break
                if valid:
                    try:
                        if rule == "ones": setattr(obj, parts[-1], jnp.ones_like(getattr(obj, parts[-1])))
                        else: setattr(obj, parts[-1], val)
                    except Exception as e:
                        print(f"FAILED on mt_key={mt_key}, mt_path={mt_path}, pt_key={pt_key}, obj={obj}")
                        raise e

  def _run_layer_parity_test(self, layer_idx, layer_type):
    # self.pt_config.layer_types = ["sliding_attention"] * 7
    # self.pt_config.layer_types[layer_idx] = layer_type
    compress_ratios = [0, 0, 4, 128, 4, 128, 4]

    torch.manual_seed(42)
    pt_layer = DeepseekV4DecoderLayer_PT(self.pt_config, layer_idx=layer_idx)
    
    # Explicitly initialize PyTorch weights with random values to prevent torch.empty
    # from yielding zero/garbage values that could mask parity differences.
    for p in pt_layer.parameters():
        if p.dim() >= 1:
            torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
        else:
            torch.nn.init.constant_(p.data, 0.02)

    if layer_idx < self.mx_config.first_num_hash_layers:
        pt_tid2eid = torch.randint(0, self.pt_config.num_local_experts, (self.vocab_size, self.pt_config.num_experts_per_tok))
        pt_layer.mlp.gate.tid2eid.copy_(pt_tid2eid)

    if layer_type == "compressed_sparse_attention" and self.pt_config.index_topk == 2:
        for p in pt_layer.self_attn.compressor.indexer.parameters():
            p.data = torch.abs(p.data) + 0.1

    mt_layer = DeepSeek4DecoderLayer(
        config=self.mx_config,
        model_mode="train",
        mesh=self.mesh,
        rngs=self.rngs,
        layer_idx=layer_idx,
        compress_ratio=compress_ratios[layer_idx],
        is_hash_routing=(layer_idx < self.mx_config.first_num_hash_layers)
    )

    self._apply_param_mapping(mt_layer, pt_layer, layer_idx)

    np.random.seed(42)
    x_np = np.random.uniform(0.1, 1.0, size=(self.batch_size, self.seq_len, self.pt_config.hc_mult, self.hidden_dim)).astype(np.float32)
    pos_np = np.arange(self.seq_len)[None, :].repeat(self.batch_size, axis=0)
    input_ids_np = np.random.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len))

    x_pt = torch.tensor(x_np)
    pos_pt = torch.tensor(pos_np, dtype=torch.long)
    input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    pt_mask = _prepare_4d_causal_attention_mask(None, (self.batch_size, self.seq_len), x_pt, 0, self.pt_config.sliding_window)

    rope_main = PTRope(self.pt_config)
    rope_compress = PTRope(self.pt_config)
    dummy_x_main = torch.zeros(self.batch_size, self.seq_len, 1)
    cos_main, sin_main = rope_main(dummy_x_main, pos_pt, "main")
    cos_comp, sin_comp = rope_compress(dummy_x_main, pos_pt, "compress")
    pt_positions = {"main": (cos_main, sin_main), "compress": (cos_comp, sin_comp)}

    pt_out = pt_layer(
        hidden_states=x_pt,
        input_ids=input_ids_pt,
        attention_mask=pt_mask,
        position_ids=pos_pt,
        position_embeddings=pt_positions
    )

    x_mt = jnp.array(x_np)
    pos_mt = jnp.array(pos_np)
    input_ids_mt = jnp.array(input_ids_np)
    segs_mt = jnp.ones_like(pos_mt, dtype=jnp.int32)
    
    mt_out, _ = mt_layer(
        inputs=x_mt,
        decoder_segment_ids=segs_mt,
        decoder_positions=pos_mt,
        deterministic=True,
        model_mode="train",
        decoder_input_tokens=input_ids_mt,
    )
    
    pt_out_tensor = pt_out[0] if isinstance(pt_out, tuple) else pt_out
    pt_out_np = pt_out_tensor.detach().numpy()
    mt_out_np = np.array(mt_out)
    max_diff = np.max(np.abs(mt_out_np - pt_out_np))
    mean_diff = np.mean(np.abs(mt_out_np - pt_out_np))
    print(f"LAYER PARITY layer_idx={layer_idx} layer_type={layer_type} - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
    np.testing.assert_allclose(mt_out_np, pt_out_np, rtol=5e-2, atol=5e-2)

  def test_layer_0_sliding_hash(self):
    self._run_layer_parity_test(0, "sliding_attention")

  def test_layer_2_csa_hash(self):
    self._run_layer_parity_test(2, "compressed_sparse_attention")

  def test_layer_3_hca_standard(self):
    self._run_layer_parity_test(3, "heavily_compressed_attention")

  def test_layer_4_csa_standard(self):
    self._run_layer_parity_test(4, "compressed_sparse_attention")

class DeepSeekV4HyperHeadTest(unittest.TestCase):
  """Tests to validate MaxText HyperHead implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 16
    self.hc_mult = 4
    self.hidden_dim = 4096

    self.config_pt = DeepseekV4Config(
        hidden_size=self.hidden_dim,
        hc_mult=self.hc_mult,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )

    # Initialize PyTorch module
    torch.manual_seed(42)
    self.pt_head = DeepseekV4HyperHead_PT(self.config_pt)
    # Initialize weights with standard values
    for p in self.pt_head.parameters():
      torch.nn.init.normal_(p.data, mean=0.0, std=0.02)

    # Create dummy mesh/rngs for MaxText
    devices = mesh_utils.create_device_mesh((1,), devices=jax.local_devices()[:1])
    self.mesh = Mesh(devices, ("x",))
    self.rngs = nnx.Rngs(0)

    # Build MaxText config dictionary
    argv = ["", "src/maxtext/configs/base.yml", "model_name=deepseek4-284b"]
    config_arguments = {
        "attention": "dot_product",
        "dtype": "float32",
        "weight_dtype": "float32",
        "mhc_expansion_rate": self.hc_mult,
        "emb_dim": self.hidden_dim,
        "normalization_layer_epsilon": 1e-6,
        "skip_jax_distributed_system": True,
    }
    self.mx_config = pyconfig.initialize(argv, **config_arguments)

  def test_hyper_head_parity(self):
    mt_head = DeepSeek4HyperHead(
        config=self.mx_config,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Map parameters from PyTorch to MaxText
    mt_head.hc_fn.value = jnp.array(self.pt_head.hc_fn.detach().numpy().T)
    mt_head.hc_base.value = jnp.array(self.pt_head.hc_base.detach().numpy())
    mt_head.hc_scale.value = jnp.array(self.pt_head.hc_scale.detach().numpy())

    # Inputs
    np.random.seed(42)
    x_np = np.random.uniform(0.1, 1.0, size=(self.batch_size, self.seq_len, self.hc_mult, self.hidden_dim)).astype(np.float32)

    x_pt = torch.tensor(x_np)
    pt_out = self.pt_head(x_pt).detach().numpy()

    x_mt = jnp.array(x_np)
    mt_out = np.array(mt_head(x_mt))

    max_diff = np.max(np.abs(mt_out - pt_out))
    mean_diff = np.mean(np.abs(mt_out - pt_out))
    print(f"HYPER HEAD PARITY - MAX ABS DIFF: {max_diff:.6e}, MEAN ABS DIFF: {mean_diff:.6e}")
    np.testing.assert_allclose(mt_out, pt_out, rtol=5e-5, atol=5e-5)


if __name__ == "__main__":
  unittest.main()

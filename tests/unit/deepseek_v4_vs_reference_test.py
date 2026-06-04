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
    apply_rotary_pos_emb as ref_apply_rotary_pos_emb,
)

from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.linears import DeepSeekV4GroupedLinear
from flax import nnx

# ==============================================================================
# Tests
# ==============================================================================


class DeepSeekV4RotaryEmbeddingTest(unittest.TestCase):
  """Tests to validate MaxText RoPE implementation against PyTorch reference."""

  def setUp(self):
    self.batch_size = 2
    self.seq_len = 16
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
    mt_rope = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=self.partial_rotary_factor,
        rope_theta=expected_theta,
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
    np.testing.assert_allclose(np.array(mt_cos), ref_cos.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(mt_sin), ref_sin.numpy(), rtol=1e-5, atol=1e-5)

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
    np.testing.assert_allclose(mt_rotated_np, ref_rotated_np, rtol=1e-5, atol=1e-5)
    print(f"Rotary Embedding test ({layer_type}) passed successfully.")


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
    np.testing.assert_allclose(np.array(mt_out), ref_out.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Grouped Linear test passed successfully.")


if __name__ == "__main__":
  unittest.main()

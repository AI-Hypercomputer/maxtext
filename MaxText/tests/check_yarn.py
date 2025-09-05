# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
python -m MaxText.tests.check_yarn
"""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx
import math


# Assuming the following imports are available from your project structure
# You may need to adjust the paths based on your file locations.
from MaxText.layers.embeddings import YarnRotaryEmbedding, GptOssRotaryEmbedding
from MaxText.common_types import DecoderBlockType

class TestRotaryEmbeddingEquivalence(unittest.TestCase):
  """
  Tests that YarnRotaryEmbedding (in GPT-OSS mode) is numerically equivalent
  to GptOssRotaryEmbedding.

  This test validates that the refactored YarnRotaryEmbedding correctly
  handles the 'concatenated' data format used by GPT-OSS, making it a
  direct replacement.
  """

  def setUp(self):
    """Set up common parameters and instantiate embedding modules."""
    # Common parameters for both embedding models.
    # Using identical parameters is crucial for a valid comparison.
    self.embedding_dims = 128
    self.max_position_embeddings = 8192
    self.original_max_position_embeddings = 4096
    self.beta_fast = 32
    self.beta_slow = 1
    # Use parameters from GptOssRotaryEmbedding defaults for consistency
    self.rope_theta = 150000.0
    self.rope_factor = 32.0
    # self.fprop_dtype = jnp.bfloat16
    self.fprop_dtype = jnp.float32

    # Batch and sequence dimensions for the input tensor
    self.batch_size = 1
    self.seq_len = 4096
    self.num_heads = 8

    # Instantiate YarnRotaryEmbedding with the specific GPT_OSS decoder block type
    self.yarn_embedding_as_gpt_oss = YarnRotaryEmbedding(
        embedding_dims=self.embedding_dims,
        max_position_embeddings=self.max_position_embeddings,
        original_max_position_embeddings=self.original_max_position_embeddings,
        beta_fast=self.beta_fast,
        beta_slow=self.beta_slow,
        rope_theta=self.rope_theta,
        rope_factor=self.rope_factor,
        fprop_dtype=self.fprop_dtype,
        truncate=False,
        attention_scaling=True,
        interleave=False,
    )

    # Instantiate the reference GptOssRotaryEmbedding
    self.gpt_oss_embedding = GptOssRotaryEmbedding(
        embedding_dims=self.embedding_dims,
        max_position_embeddings=self.max_position_embeddings,
        original_max_position_embeddings=self.original_max_position_embeddings,
        beta_fast=self.beta_fast,
        beta_slow=self.beta_slow,
        rope_theta=self.rope_theta,
        rope_factor=self.rope_factor,
        fprop_dtype=self.fprop_dtype,
        truncate=False,
        rngs=nnx.Rngs(0),  # Pass dummy RNGs
    )

    # Create dummy input data and positions
    key = jax.random.PRNGKey(0)
    # Input shape is (batch, sequence_length, num_heads, embedding_dims)
    self.inputs = jax.random.normal(
        key, (self.batch_size, self.seq_len, self.num_heads, self.embedding_dims), dtype=jnp.float32
    )
    # Position shape is (batch, sequence_length)
    self.position = jnp.arange(self.seq_len, dtype=jnp.int32)[None, :]
    self.position = jnp.tile(self.position, (self.batch_size, 1))

  def test_frequency_calculation_equivalence(self):
    """
    Validates that the frequency generation logic is identical between the two modules.
    """
    # 1. Get the reference (sin, cos) tuple from the GPT-OSS implementation
    sin_ref, cos_ref = self.gpt_oss_embedding._generate_pos_embeddings(self.position)

    # 2. Replicate the logic using the Yarn implementation's freqs_cis property

    # Get the unscaled complex frequencies for each position
    freqs_complex = self.yarn_embedding_as_gpt_oss.freqs_cis[self.position]

    # Separate into unscaled sin and cos
    sin_yarn_unscaled = jnp.imag(freqs_complex)
    cos_yarn_unscaled = jnp.real(freqs_complex)

    # Manually apply the same scaling factor
    # NOTE: Ensure self.rope_factor is set correctly in setUp()
    rope_factor = self.yarn_embedding_as_gpt_oss.rope_factor
    attention_scaling = 1.0 if rope_factor <= 1 else (0.1 * math.log(rope_factor) + 1.0)

    sin_yarn_scaled = sin_yarn_unscaled * attention_scaling
    cos_yarn_scaled = cos_yarn_unscaled * attention_scaling

    print("\n==sin_ref\n", sin_ref)
    print("\n==sin_yarn_scaled\n", sin_yarn_scaled)

    # Calculate the absolute difference between the two arrays
    absolute_difference = jnp.abs(sin_ref - sin_yarn_scaled)
    max_diff = jnp.max(absolute_difference)

    # Print the maximum difference to see how large the error is
    print(f"\nMaximum absolute difference between sin arrays: {max_diff}\n")

    # 3. Assert that both are now identical
    self.assertTrue(
        jnp.allclose(sin_ref, sin_yarn_scaled, atol=1e-5, rtol=1e-5), "Sin values from frequency calculations do not match."
    )
    self.assertTrue(
        jnp.allclose(cos_ref, cos_yarn_scaled, atol=1e-5, rtol=1e-5), "Cos values from frequency calculations do not match."
    )
    print("Test passed: Underlying frequency calculations are equivalent.")

  def test_direct_equivalence(self):
    """
    Validates that both modules produce identical output given the same input,
    confirming the internal data format handling is correct.
    """
    # a = self.gpt_oss_embedding._generate_pos_embeddings(self.position)
    # b=self.yarn_embedding_as_gpt_oss.freqs_cis[self.position]
    # jnp.allclose(a, b, atol=1e-3, rtol=1e-3)

    # 1. Get the output from the reference GptOssRotaryEmbedding
    gpt_oss_output = self.gpt_oss_embedding(self.inputs, self.position)

    # 2. Get the output from the modified YarnRotaryEmbedding
    # No data conversion is needed as it now handles the format internally.
    yarn_output = self.yarn_embedding_as_gpt_oss(self.inputs, self.position)

    # 3. Assert that the two outputs are numerically very close.
    # A small tolerance (atol) is used to account for minor floating-point differences.
    print(f"\n==gpt_oss_output==\nmean{gpt_oss_output.mean()}", gpt_oss_output)
    print(f"\n==yarn_output==\nmean{yarn_output.mean()}", yarn_output)

    # Calculate the absolute difference between the two arrays
    absolute_difference = jnp.abs(gpt_oss_output - yarn_output)
    max_diff = jnp.max(absolute_difference)
    # Print the maximum difference to see how large the error is
    print(f"\nMaximum absolute difference between output: {max_diff}\n")
    self.assertTrue(
        jnp.allclose(gpt_oss_output, yarn_output, atol=1e-5, rtol=1e-5),
        "Outputs from GptOssRotaryEmbedding and the refactored YarnRotaryEmbedding (GPT_OSS mode) should be identical.",
    )
    print("Test passed: Outputs are functionally equivalent.")


# Boilerplate to run the tests in a script or interactive environment
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

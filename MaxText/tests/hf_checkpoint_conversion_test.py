"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Tests for kernels """

import numpy as np
import jax.numpy as jnp
from MaxText.max_utils import permute_to_match_maxtext_rope, unpermute_from_match_maxtext_rope
from MaxText.ckpt_conversion.utils.utils import convert_jax_weight_to_numpy
import unittest


class HFCheckpointConversionTest(unittest.TestCase):

  def test_huggingface_to_maxtext_back_to_huggingface_flow(self):
    base_num_query_heads = 16
    head_dim = 32
    wq = np.arange(base_num_query_heads * head_dim * base_num_query_heads * head_dim, dtype=np.float16).reshape(
        base_num_query_heads * head_dim, base_num_query_heads * head_dim
    )
    wq1 = wq.transpose()
    wq2 = np.reshape(wq1, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

    wq3 = permute_to_match_maxtext_rope(wq2)
    stack_shape = (1,)
    x = np.zeros(stack_shape + wq3.shape, dtype=np.float16)
    x[0, ...] = wq3
    x = np.transpose(x, axes=(1, 0, 2, 3))

    x = x[:, 0, :, :]
    wq4 = unpermute_from_match_maxtext_rope(x, "llama3.1")
    wq5 = wq4.reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim)
    wq6 = wq5.transpose()

    self.assertTrue(np.array_equal(wq, wq6), "Test failed: wq does not match wq6")
    self.assertTrue(np.array_equal(wq1, wq5), "Test failed: wq1 does not match wq5")
    self.assertTrue(np.array_equal(wq2, wq4), "Test failed: wq2 does not match wq4")

  def test_convert_jax_weight_to_numpy(self):
    # Case 1: Basic conversion (float32 JAX array)
    jax_arr_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    np_arr_f32 = convert_jax_weight_to_numpy(jax_arr_f32)
    self.assertIsInstance(np_arr_f32, np.ndarray, "Output should be a NumPy array")
    self.assertEqual(np_arr_f32.dtype, np.float32, "Output dtype should be float32")
    np.testing.assert_array_equal(np_arr_f32, np.array([1.0, 2.0, 3.0], dtype=np.float32), "Array values should match")
    self.assertEqual(np_arr_f32.shape, jax_arr_f32.shape, "Shapes should match")

    # Case 2: Conversion with specified output dtype (JAX float32 to NumPy float16)
    np_arr_f16_specified = convert_jax_weight_to_numpy(jax_arr_f32, dtype_str="float16")
    self.assertEqual(np_arr_f16_specified.dtype, np.float16, "Output dtype should be float16 when specified")
    np.testing.assert_array_equal(
        np_arr_f16_specified, np.array([1.0, 2.0, 3.0], dtype=np.float16), "Array values should match after dtype conversion"
    )
    self.assertEqual(np_arr_f16_specified.shape, jax_arr_f32.shape, "Shapes should match")

    # Case 3: Conversion of bfloat16 JAX array (implicitly to NumPy bfloat16 if supported, or float32)
    jax_arr_bf16 = jnp.array([4.0, 5.0, 6.0], dtype=jnp.bfloat16)
    np_arr_bf16 = convert_jax_weight_to_numpy(jax_arr_bf16)
    # Check if the output dtype is bfloat16 (if supported) or float32
    self.assertIn(
        np_arr_bf16.dtype,
        [np.dtype("bfloat16"), np.float32],
        "Output dtype for bfloat16 input should be bfloat16 or float32",
    )
    # Compare values after potentially casting expected to the actual output dtype for robust comparison
    expected_bf16_values = np.array([4.0, 5.0, 6.0]).astype(np_arr_bf16.dtype)
    np.testing.assert_array_equal(np_arr_bf16, expected_bf16_values, "Array values should match for bfloat16 input")
    self.assertEqual(np_arr_bf16.shape, jax_arr_bf16.shape, "Shapes should match")

    # Case 4: Multi-dimensional array
    jax_md_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    np_md_arr = convert_jax_weight_to_numpy(jax_md_arr)
    self.assertEqual(np_md_arr.shape, jax_md_arr.shape, "Multi-dimensional shapes should match")
    np.testing.assert_array_equal(
        np_md_arr, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "Multi-dimensional array values should match"
    )


if __name__ == "__main__":
  unittest.main()

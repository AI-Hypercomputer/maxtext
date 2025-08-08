# SPDX-License-Identifier: Apache-2.0

""" Tests for kernels """

import numpy as np
from MaxText.max_utils import permute_to_match_maxtext_rope, unpermute_from_match_maxtext_rope
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

    if not np.array_equal(wq, wq6):
      print("Test failed: wq does not match wq6")

    if not np.array_equal(wq1, wq5):
      print("Test failed: wq1 does not match wq5")

    if not np.array_equal(wq2, wq4):
      print("Test failed: wq2 does not match wq4")


if __name__ == "__main__":
  unittest.main()

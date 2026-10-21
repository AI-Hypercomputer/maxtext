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

"""Correctness tests for the SparseCore ragged gather-reduce kernel (ragged_gather_reduce_v2).

The kernel output is compared against the pure-JAX `_fallback_implementation`

Non-power-of-2 group sizes (e.g. 6) exercise the unsigned-division lowering of the destination row index.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import ragged_gather_reduce_v2


_HIDDEN = 7168
_NUM_ROWS = 4096


def _make_inputs(seed, *, num_rows, hidden, num_groups, group_size):
  """Random gather-reduce inputs."""
  k_x, k_idx, k_w, k_mask = jax.random.split(jax.random.key(seed), 4)
  input_size = num_groups * group_size
  x = jax.random.normal(k_x, (num_rows, hidden), jnp.bfloat16)
  indices = jax.random.randint(k_idx, (input_size,), 0, num_rows, jnp.int32)
  topk_weights = jax.random.uniform(k_w, (input_size,), jnp.bfloat16)
  valid = jax.random.bernoulli(k_mask, 0.7, (input_size,))
  return x, indices, topk_weights, valid


@pytest.mark.tpu_only
class RaggedGatherReduceV2Test(parameterized.TestCase):
  """SparseCore ragged_gather_reduce vs _fallback_implementation."""

  def setUp(self):
    super().setUp()
    if not any(d.platform == "tpu" for d in jax.devices()):
      self.skipTest("The SparseCore ragged gather-reduce kernel requires TPU.")
    if pltpu.get_tpu_info().sparse_core is None:
      self.skipTest("The ragged gather-reduce kernel requires a TPU with SparseCore.")

  def _assert_matches_fallback(self, *, group_size, num_groups):
    """Checks that the SparseCore kernel output matches the fallback implementation."""
    x, indices, topk_weights, valid = _make_inputs(
        0,
        num_rows=_NUM_ROWS,
        hidden=_HIDDEN,
        num_groups=num_groups,
        group_size=group_size,
    )
    kernel = jax.jit(functools.partial(ragged_gather_reduce_v2.ragged_gather_reduce, reduce_group_size=group_size))
    reference = jax.jit(functools.partial(ragged_gather_reduce_v2._fallback_implementation, reduce_group_size=group_size))  # pylint: disable=protected-access
    actual = kernel(x, indices, topk_weights, valid)
    expected = reference(x, indices, topk_weights, valid)

    self.assertEqual(actual.dtype, expected.dtype)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), rtol=1e-2, atol=1e-2)

  @parameterized.parameters(4, 6, 8)
  def test_matches_fallback(self, group_size):
    self._assert_matches_fallback(group_size=group_size, num_groups=256)


if __name__ == "__main__":
  absltest.main()

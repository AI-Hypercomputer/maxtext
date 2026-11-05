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

"""ops.gmm, gmm_v2 and tgmm_v2 accept a Python int group_offset when their bodies run untraced (jax.disable_jit())."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2

TILING = (128,) * 9
GROUP_SIZES = [128, 256, 128]


def _inputs(num_groups, dtype=jnp.float32):
  lhs = jax.random.normal(jax.random.PRNGKey(0), (512, 128), jnp.float32).astype(dtype)
  rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, 128, 128), jnp.float32).astype(dtype)
  dout = jax.random.normal(jax.random.PRNGKey(2), (512, 128), jnp.float32).astype(dtype)
  return lhs, rhs, dout


def _fwd_bwd(gmm_fn, lhs, rhs, dout):
  out, vjp_fn = jax.vjp(gmm_fn, lhs, rhs)
  dlhs, drhs = vjp_fn(dout)
  return out, dlhs, drhs


class MegabloxGroupOffsetTest(unittest.TestCase):
  """Megablox backend (interpret mode, runs on CPU): forward gmm + backward gmm/tgmm under jax.disable_jit()."""

  def _check(self, group_offset, num_local_groups):
    """Checks ops.gmm fwd + bwd with a Python int group_offset: disable_jit vs jit, and vs a dense reference."""
    lhs, rhs, dout = _inputs(num_local_groups)
    group_sizes = jnp.asarray(GROUP_SIZES, jnp.int32)

    def gmm_fn(l, r):
      # MoE layers pass group_offset as a Python int when expert parallelism is off.
      return ops.gmm(l, r, group_sizes, jnp.float32, TILING, group_offset, interpret=True)

    jit_res = jax.jit(lambda l, r, d: _fwd_bwd(gmm_fn, l, r, d))(lhs, rhs, dout)
    with jax.disable_jit():
      eager_res = _fwd_bwd(gmm_fn, lhs, rhs, dout)
    for name, eager, jitted in zip(("out", "dlhs", "drhs"), eager_res, jit_res):
      np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted), err_msg=name)

    # Dense reference: rows of the local groups use their expert, other rows are zero.
    ends = np.cumsum(GROUP_SIZES)
    starts = ends - np.asarray(GROUP_SIZES)
    want = np.zeros((lhs.shape[0], rhs.shape[2]), np.float32)
    for local, g in enumerate(range(group_offset, group_offset + num_local_groups)):
      want[starts[g] : ends[g]] = np.asarray(lhs[starts[g] : ends[g]]) @ np.asarray(rhs[local])
    np.testing.assert_allclose(np.asarray(eager_res[0]), want, rtol=1e-4, atol=1e-4)

  def test_python_int_group_offset_zero(self):
    self._check(group_offset=0, num_local_groups=3)

  def test_python_int_group_offset_nonzero(self):
    # Two local groups starting at global group 1 (e.g. one expert-parallel shard).
    self._check(group_offset=1, num_local_groups=2)


class GmmV2GroupOffsetWiringTest(unittest.TestCase):
  """On CPU: under jax.disable_jit(), ops.gmm hands gmm_v2 / tgmm_v2 an int32 array group_offset.

  The kernels need a TPU, so they are replaced by fakes that record the group_offset they receive.
  GmmV2KernelTest runs the real kernels.
  """

  def test_kernels_receive_int32_array(self):
    lhs, rhs, dout = _inputs(3)
    group_sizes = jnp.asarray(GROUP_SIZES, jnp.int32)
    seen = []

    def fake_gmm_v2(lhs, rhs, group_sizes, **kwargs):
      seen.append(("gmm_v2", kwargs["group_offset"]))
      return jnp.zeros((lhs.shape[0], rhs.shape[2]), kwargs["preferred_element_type"])

    def fake_tgmm_v2(lhs, rhs, group_sizes, num_actual_groups, **kwargs):
      seen.append(("tgmm_v2", kwargs["group_offset"]))
      return jnp.zeros((num_actual_groups, lhs.shape[1], rhs.shape[1]), kwargs["preferred_element_type"])

    def gmm_fn(l, r):
      return ops.gmm(l, r, group_sizes, jnp.float32, TILING, 0, use_tokamax_backend=True, use_gmm_v2=True)

    with mock.patch.object(gmm_v2, "gmm_v2", fake_gmm_v2), mock.patch.object(tgmm_v2, "tgmm_v2", fake_tgmm_v2):
      with jax.disable_jit():
        _fwd_bwd(gmm_fn, lhs, rhs, dout)

    self.assertEqual([name for name, _ in seen], ["gmm_v2", "gmm_v2", "tgmm_v2"])  # fwd, dlhs, drhs
    for name, group_offset in seen:
      self.assertIsInstance(group_offset, jax.Array, msg=name)
      self.assertEqual(group_offset.dtype, jnp.int32, msg=name)
      self.assertEqual(int(group_offset), 0, msg=name)

  def test_validate_tgmm_inputs_accepts_python_int(self):
    group_sizes = jnp.asarray(GROUP_SIZES, jnp.int32)
    for group_offset in (None, 0, 1, np.int32(1), jnp.int32(1), jnp.ones((1,), jnp.int32)):
      tgmm_v2.validate_tgmm_inputs(group_sizes, num_actual_groups=2, group_offset=group_offset)
    with self.assertRaises(ValueError):
      tgmm_v2.validate_tgmm_inputs(group_sizes, num_actual_groups=2, group_offset=2)


@pytest.mark.tpu_only
class GmmV2KernelTest(unittest.TestCase):
  """TPU: gmm_v2 / tgmm_v2 accept group_offset as a Python int, a 0-d array or an int32[1] array.

  The kernel bodies are traced inline (`__wrapped__` inside an outer jit), so a Python int reaches
  the group_offset normalization untraced, as it does under jax.disable_jit().
  """

  def test_group_offset_forms_match(self):
    lhs, rhs, dout = _inputs(2, jnp.bfloat16)
    group_sizes = jnp.asarray(GROUP_SIZES, jnp.int32)
    tiles = gmm_v2.TileSizes(tile_m=128, tile_k=128, tile_n=128)
    results = []
    for group_offset in (1, np.int32(1), jnp.int32(1), jnp.ones((1,), jnp.int32)):

      def run(l, r, d, go=group_offset):
        out = gmm_v2.gmm_v2.__wrapped__(l, r, group_sizes, group_offset=go, tile_info=tiles)
        drhs = tgmm_v2.tgmm_v2.__wrapped__(l, d, group_sizes, 2, group_offset=go, tile_info=tiles)
        return out, drhs

      results.append(jax.jit(run)(lhs, rhs, dout))
    for out, drhs in results[1:]:
      np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(results[0][0], np.float32))
      np.testing.assert_array_equal(np.asarray(drhs, np.float32), np.asarray(results[0][1], np.float32))


if __name__ == "__main__":
  unittest.main()

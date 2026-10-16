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

"""TPU tests: the SwiGLU in the gmm_v2 epilogue (megablox ops.gmm_silu, moe_wi_gmm_fused_swiglu) against the unfused GMM.

gmm_v2(fuse_act="silu", return_pre_act=True) returns (act, gate, up) from one accumulator: gate and up must be the
unfused gmm_v2 output's halves bit for bit (they are rounded the same way), every row outside the computed groups must
be zero in all three, and act must be silu(gate) * up computed from the float32 accumulator (close to, not equal to,
the same activation of the bf16-rounded halves: that is the numerics change). The act-only call must give the same
act. ops.gmm_silu's VJP is the unfused one: its gradients must equal, value for value, the VJP of the unfused path
(ops.gmm, then silu(gate) * up on its bf16 output). Groups are skewed, with an empty group, padding rows past the
last group, boundaries inside a sublane, and an expert-parallel shard (group_offset).

Pallas TPU kernels: these run only on a TPU. On CPU the moe.py wiring of the key is covered by
moe_wi_gmm_fused_swiglu_test.py with a jnp stand-in for the kernel.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2

pytestmark = pytest.mark.tpu_only  # Pallas TPU kernels.

M, K, N = 1024, 512, 512  # rhs is [groups, K, 2 * N] = [gate | up]
GROUP_SIZES = np.array([300, 0, 517, 133], np.int32)  # sum 950 < M: padding rows; group ends inside sublanes
TILING = (256, 512, 1024, 256, 512, 512, 256, 512, 512)  # gmm's 9-tuple; gmm_silu halves the forward tile_n


def _rand(key, shape, scale=1.0):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(jnp.bfloat16)


def _reference_accumulator(lhs, rhs, group_sizes, group_offset):
  """f32 lhs @ rhs[expert of row] for the rows of the local groups [group_offset, group_offset + groups), else 0."""
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= ends[None, :], axis=1) - group_offset
  out = jnp.zeros((lhs.shape[0], rhs.shape[-1]), jnp.float32)
  for e in range(rhs.shape[0]):
    out = out + jnp.where((local == e)[:, None], lhs.astype(jnp.float32) @ rhs[e].astype(jnp.float32), 0.0)
  return out


def _f32(x):
  return np.asarray(x.astype(jnp.float32))


class GmmV2PreActTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")

  @parameterized.product(group_offset=[0, 2], tile_k=[256, 512])
  def test_pre_act_halves_are_the_unfused_output(self, group_offset, tile_k):
    k0, k1 = jax.random.split(jax.random.key(group_offset * 7 + tile_k))
    lhs = _rand(k0, (M, K))
    rhs = _rand(k1, (len(GROUP_SIZES) - group_offset, K, 2 * N), 0.05)
    gs, go = jnp.asarray(GROUP_SIZES), jnp.array([group_offset], jnp.int32)
    unfused = jax.jit(
        functools.partial(
            gmm_v2.gmm_v2, tile_info=gmm_v2.TileSizes(256, tile_k, 512), preferred_element_type=jnp.bfloat16
        )
    )(lhs, rhs, gs, group_offset=go)
    fused = functools.partial(
        gmm_v2.gmm_v2, tile_info=gmm_v2.TileSizes(256, tile_k, 256), preferred_element_type=jnp.bfloat16, fuse_act="silu"
    )
    act, gate, up = jax.jit(functools.partial(fused, return_pre_act=True))(lhs, rhs, gs, group_offset=go)
    act_only = jax.jit(fused)(lhs, rhs, gs, group_offset=go)

    np.testing.assert_array_equal(_f32(gate), _f32(unfused[:, :N]))
    np.testing.assert_array_equal(_f32(up), _f32(unfused[:, N:]))
    np.testing.assert_array_equal(_f32(act_only), _f32(act))
    # Rows outside the local groups: zero in every output.
    computed = np.zeros(M, bool)
    start = int(np.sum(GROUP_SIZES[:group_offset]))
    computed[start : start + int(np.sum(GROUP_SIZES[group_offset:]))] = True
    for out in (act, gate, up):
      self.assertTrue(np.all(_f32(out)[~computed] == 0))
    # act is silu on the f32 accumulator, rounded once to bf16.
    acc = _reference_accumulator(lhs, rhs, gs, group_offset)
    want = (jax.nn.silu(acc[:, :N]) * acc[:, N:]).astype(jnp.bfloat16)
    np.testing.assert_allclose(_f32(act), _f32(want), rtol=1e-2, atol=1e-3)


class GmmSiluTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")

  @parameterized.parameters(0, 2)
  def test_forward_and_vjp_match_the_unfused_path(self, group_offset):
    k0, k1, k2 = jax.random.split(jax.random.key(group_offset), 3)
    lhs = _rand(k0, (M, K))
    rhs = _rand(k1, (len(GROUP_SIZES) - group_offset, K, 2 * N), 0.05)
    cot = _rand(k2, (M, N))
    gs, go = jnp.asarray(GROUP_SIZES), jnp.array([group_offset], jnp.int32)

    def unfused(lhs, rhs):
      out = ops.gmm(
          lhs,
          rhs,
          gs,
          preferred_element_type=jnp.bfloat16,
          tiling=TILING,
          group_offset=go,
          use_tokamax_backend=True,
          use_gmm_v2=True,
      )
      return (jax.nn.silu(out[:, :N]) * out[:, N:]).astype(jnp.bfloat16)

    def fused(lhs, rhs):
      return ops.gmm_silu(lhs, rhs, gs, preferred_element_type=jnp.bfloat16, tiling=TILING, group_offset=go)

    @jax.jit
    def run(lhs, rhs, cot):
      y_u, vjp_u = jax.vjp(unfused, lhs, rhs)
      y_f, vjp_f = jax.vjp(fused, lhs, rhs)
      return y_u, vjp_u(cot), y_f, vjp_f(cot)

    y_u, (dl_u, dr_u), y_f, (dl_f, dr_f) = run(lhs, rhs, cot)
    # Forward: the f32-accumulator activation, close to the unfused one.
    np.testing.assert_allclose(_f32(y_f), _f32(y_u), rtol=2e-2, atol=2e-3)
    # Backward: the unfused backward on the same bf16 gate and up, so the same values.
    np.testing.assert_array_equal(_f32(dl_f), _f32(dl_u))
    np.testing.assert_array_equal(_f32(dr_f), _f32(dr_u))


if __name__ == "__main__":
  absltest.main()

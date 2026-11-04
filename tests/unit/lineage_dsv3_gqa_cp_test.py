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

"""CPU equivalence test for the GQA Ulysses (all-to-all) context parallelism.

Compares `dsv3_mla._gqa_qkv` / `_gqa_out_projection` with head-sharded weights
(all-gather of x + output reduce-scatter) against head-replicated weights
(`_gqa_qkv_a2a` all-to-all path), forward and backward, on an 8-device CPU
mesh with a 4-way sequence (CP) axis.
"""

import jax

jax.config.update("jax_num_cpu_devices", 8)

# pylint: disable=g-import-not-at-top
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest

from maxtext.experimental.lineage import dsv3_mla

# pylint: enable=g-import-not-at-top

P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding

_B, _T, _D = 2, 64, 32
_NQ, _NKV, _H = 8, 2, 16


def _attention(q, k, v):
  """Causal MHA on [B, T, N, H] inputs, sharding-agnostic reference."""
  s = jnp.einsum("bqnh,bknh->bnqk", q, k)
  t = q.shape[1]
  mask = jnp.tril(jnp.ones((t, t), dtype=bool))
  s = jnp.where(mask[None, None], s, -1e30)
  p = jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(q.dtype)
  return jnp.einsum("bnqk,bknh->bqnh", p, v)


class GqaCpA2aTest(absltest.TestCase):

  def test_a2a_matches_all_gather(self):
    devices = np.array(jax.devices()[:8]).reshape(2, 4)
    mesh = jax.sharding.Mesh(devices, ("b", "s"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
    rng = np.random.default_rng(0)

    def arr(shape, spec, scale=0.1):
      return jax.device_put(
          jnp.asarray(rng.normal(size=shape) * scale, jnp.float32),
          NamedSharding(mesh, spec),
      )

    x = arr((_B, _T, _D), P("b", "s", None), 1.0)
    wq_val = rng.normal(size=(_D, _NQ, _H)) * 0.1
    wkv_val = rng.normal(size=(_D, 2 * _NKV * _H)) * 0.1
    wo_val = rng.normal(size=(_NQ, _H, _D)) * 0.1
    pos = np.arange(_T)[None, :, None, None] / 10.0
    freqs = np.arange(_H)[None, None, None, :] / _H
    cos = jax.device_put(
        jnp.asarray(np.broadcast_to(np.cos(pos * freqs), (_B, _T, 1, _H))),
        NamedSharding(mesh, P("b", None, None, None)),
    )
    sin = jax.device_put(
        jnp.asarray(np.broadcast_to(np.sin(pos * freqs), (_B, _T, 1, _H))),
        NamedSharding(mesh, P("b", None, None, None)),
    )

    def put(val, spec):
      return jax.device_put(jnp.asarray(val, jnp.float32), NamedSharding(mesh, spec))

    def loss_and_out(x, wq, wkv, wo):
      q, k, v = dsv3_mla._gqa_qkv(x, wq, wkv, (cos, sin), mesh=mesh)
      o = _attention(q, k, v)
      o = jax.sharding.reshard(o, P("b", None, "s", None))
      out = dsv3_mla._gqa_out_projection(o, wo)
      return jnp.sum(out * out), out

    grad_fn = jax.jit(jax.value_and_grad(loss_and_out, argnums=(0, 1, 2, 3), has_aux=True))
    with jax.set_mesh(mesh):
      # Head-sharded Wq / Wo: all-gather x + reduce-scatter path.
      (l_ag, out_ag), g_ag = grad_fn(
          x,
          put(wq_val, P(None, "s", None)),
          put(wkv_val, P(None, None)),
          put(wo_val, P("s", None, None)),
      )
      # Head-replicated Wq / Wo: Ulysses all-to-all path.
      (l_a2a, out_a2a), g_a2a = grad_fn(
          x,
          put(wq_val, P(None, None, None)),
          put(wkv_val, P(None, None)),
          put(wo_val, P(None, None, None)),
      )

    self.assertEqual(jax.typeof(out_a2a).sharding.spec, P("b", "s", None))
    np.testing.assert_allclose(l_a2a, l_ag, rtol=1e-4)
    np.testing.assert_allclose(out_a2a, out_ag, rtol=1e-4, atol=1e-5)
    for name, a, b in zip(("x", "wq", "wkv", "wo"), g_a2a, g_ag):
      with self.subTest(name):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  absltest.main()

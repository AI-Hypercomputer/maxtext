# Copyright 2025-2026 Google LLC
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

"""Numerics of the OLMoE3 small-op cleanup flags vs their default paths (CPU, Pallas interpret mode).

- kda_flat_head_layout: KDA module output and gradients match the token-major fused path.
- head_norm_flat: Pallas per-head L2/RMS normalization matches the reshape reference.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.kernels import head_norm
from maxtext.models import olmoe3
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR


def _config(extra=()):
  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "model_name=olmoe3-30m",
          "override_model_config=true",
          "run_name=olmoe3_smallops_test",
          "enable_checkpointing=False",
          "per_device_batch_size=1",
          "max_target_length=256",
          "dtype=float32",
          "weight_dtype=float32",
          "gdn_num_key_heads=2",
          "gdn_num_value_heads=2",
          "gdn_key_head_dim=128",
          "gdn_value_head_dim=256",
          "gdn_chunk_size=64",
          "gdn_state_dtype=float32",
          "kda_chunked_impl=subblock",
          "kda_sub_block=16",
          "kda_pallas_fused=true",
          "kda_fused_token_major=true",
          "kda_inverse_precision=highest",
          "kda_fused_input_proj=true",
          "skip_jax_distributed_system=True",
          *extra,
      ],
  )


def _rel(a, b):
  a = np.asarray(a, np.float32)
  b = np.asarray(b, np.float32)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


class KdaFlatHeadLayoutTest(parameterized.TestCase):

  @parameterized.named_parameters(("no_segments", False), ("packed_segments", True))
  def test_flat_matches_default(self, packed):
    cfg_ref = _config()
    cfg_flat = _config(("kda_flat_head_layout=true",))
    mesh = maxtext_utils.get_mesh_from_config(cfg_ref)
    with nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      ref = olmoe3.OLMoE3KimiDeltaAttention(cfg_ref, mesh, rngs=nnx.Rngs(0))
      flat = olmoe3.OLMoE3KimiDeltaAttention(cfg_flat, mesh, rngs=nnx.Rngs(0))
    self.assertTrue(flat._use_flat_head_layout(256))  # pylint: disable=protected-access
    self.assertFalse(ref._use_flat_head_layout(256))  # pylint: disable=protected-access
    # Non-trivial per-head parameters (A_log differs per head; o_norm gain is not all ones).
    _, ref_params = nnx.split(ref, nnx.Param)
    _, flat_params = nnx.split(flat, nnx.Param)
    k = jax.random.PRNGKey(1)
    x = jax.random.normal(k, (1, 256, cfg_ref.emb_dim), jnp.float32)
    seg = None
    if packed:
      seg = jnp.concatenate([jnp.ones((1, 100), jnp.int32), 2 * jnp.ones((1, 156), jnp.int32)], axis=1)
    ref_params = jax.tree.map(lambda p: p + 0.05 * jax.random.normal(jax.random.PRNGKey(7), p.shape, p.dtype), ref_params)
    flat_params = ref_params

    def run(module, params, x):
      graphdef, _ = nnx.split(module, nnx.Param)
      m = nnx.merge(graphdef, params)
      return m(x, seg)

    def loss(module, params, x):
      out = run(module, params, x)
      return jnp.sum(out * jnp.cos(jnp.arange(out.size, dtype=jnp.float32).reshape(out.shape) * 0.01))

    with nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      out_r = run(ref, ref_params, x)
      out_f = run(flat, flat_params, x)
      gr = jax.grad(lambda p, x: loss(ref, p, x), argnums=(0, 1))(ref_params, x)
      gf = jax.grad(lambda p, x: loss(flat, p, x), argnums=(0, 1))(flat_params, x)
    self.assertEqual(out_r.shape, out_f.shape)
    self.assertLess(_rel(out_f, out_r), 1e-5)
    self.assertLess(_rel(gf[1], gr[1]), 1e-4)
    leaves_r = jax.tree_util.tree_leaves_with_path(gr[0])
    leaves_f = dict(jax.tree_util.tree_leaves_with_path(gf[0]))
    for path, g in leaves_r:
      self.assertLess(_rel(leaves_f[path], g), 1e-4, msg=str(path))


class HeadNormFlatTest(parameterized.TestCase):

  @parameterized.parameters((False, jnp.float32, 1.0), (False, jnp.float32, 0.0625), (True, jnp.bfloat16, 1.0))
  def test_matches_reshape_reference(self, mean, dtype, scale):
    heads, d = 3, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 40, heads * d), jnp.float32).astype(dtype)
    w = jax.random.normal(jax.random.PRNGKey(1), x.shape, jnp.float32)

    def ref(x):
      xf = x.astype(jnp.float32).reshape(2, 40, heads, d)
      s = jnp.sum(xf * xf, -1, keepdims=True)
      s = s / d if mean else s
      return (xf * jax.lax.rsqrt(s + 1e-6)).reshape(x.shape) * scale

    def fused(x):
      return head_norm.head_norm_flat(
          x, heads, mean=mean, eps=1e-6, scale=scale, out_dtype=jnp.float32, block_n=16
      )

    np.testing.assert_allclose(fused(x), ref(x), rtol=1e-5, atol=1e-5)
    gr = jax.grad(lambda x: jnp.sum(ref(x) * w))(x)
    gf = jax.grad(lambda x: jnp.sum(fused(x) * w))(x)
    self.assertEqual(gf.dtype, x.dtype)
    self.assertLess(_rel(gf, gr), 1e-2 if dtype == jnp.bfloat16 else 1e-5)


if __name__ == "__main__":
  absltest.main()

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

"""Reference tests for the MoE top-k combine (un-sort + weighted sum) paths.

Covers `RoutedMoE.unpermute` (generic gather branch) and `sort_activations.unroute` /
`sort_activations.route` (non-Mosaic path), forward and backward, against a NumPy reference.
These paths index in (k, tokens) order so the combine sums over a major axis instead of
reshaping the [tokens * k, emb] activations to [tokens, k, emb].
"""

import types
import unittest

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.kernels import sort_activations
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _routing(rng, num_tokens, k, num_experts):
  """Random top-k expert ids: k distinct experts per token."""
  return np.stack([rng.permutation(num_experts)[:k] for _ in range(num_tokens)]).astype(np.int32)


def _fake_moe(k, float32_weight_sum=False, decoder_block=ctypes.DecoderBlockType.DEEPSEEK):
  """The attributes `RoutedMoE.unpermute` reads on its generic (no ragged sort / TC routing) branch."""
  config = types.SimpleNamespace(
      use_ragged_sort=False,
      use_ring_of_experts=False,
      matmul_precision="highest",
      decoder_block=decoder_block,
      float32_weight_sum=float32_weight_sum,
  )
  return types.SimpleNamespace(config=config, num_experts_per_tok=k, dtype=jnp.float32)


class UnpermuteReferenceTest(unittest.TestCase):
  """`RoutedMoE.unpermute` matches out[t] = sum_k w[t, k] * expert_out[sorted position of (t, k)]."""

  def _check(self, batch, seq, k, num_experts, emb, use_custom_sort_vjp, float32_weight_sum=False):
    """Checks forward and backward parity of `RoutedMoE.unpermute` against a NumPy reference."""
    rng = np.random.default_rng(0)
    num_tokens = batch * seq
    selected = _routing(rng, num_tokens, k, num_experts)
    # As in RoutedMoE.permute: sort the flattened (token, k) slots by expert.
    sort_idx = np.argsort(selected.reshape(-1), kind="stable").astype(np.int32)
    intermediate = rng.standard_normal((num_tokens * k, emb)).astype(np.float32)
    weights = rng.standard_normal((batch, seq, k)).astype(np.float32)
    fake = _fake_moe(k, float32_weight_sum=float32_weight_sum)

    def f(inter, w):
      return moe.RoutedMoE.unpermute(
          fake, inter, jnp.asarray(sort_idx), w, batch, seq, use_custom_sort_vjp=use_custom_sort_vjp
      )

    # Reference: slot s = t * k + j sits at row inv[s] of `intermediate`.
    inv = np.argsort(sort_idx)
    w2 = weights.reshape(num_tokens, k)
    want = np.einsum("tke,tk->te", intermediate[inv].reshape(num_tokens, k, emb), w2).reshape(batch, seq, emb)
    got = f(jnp.asarray(intermediate), jnp.asarray(weights))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)

    # Backward vs. autodiff of a plain-jnp reference.
    cot = rng.standard_normal((batch, seq, emb)).astype(np.float32)

    def ref(inter, w):
      gathered = inter[jnp.asarray(inv)].reshape(num_tokens, k, emb)
      return jnp.einsum("tke,tk->te", gathered, w.reshape(num_tokens, k)).reshape(batch, seq, emb)

    got_g = jax.vjp(f, jnp.asarray(intermediate), jnp.asarray(weights))[1](jnp.asarray(cot))
    want_g = jax.vjp(ref, jnp.asarray(intermediate), jnp.asarray(weights))[1](jnp.asarray(cot))
    for g, w in zip(got_g, want_g):
      np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5, atol=1e-5)

  def test_custom_sort_vjp(self):
    self._check(batch=2, seq=8, k=4, num_experts=16, emb=8, use_custom_sort_vjp=True)

  def test_default_vjp(self):
    self._check(batch=2, seq=8, k=4, num_experts=16, emb=8, use_custom_sort_vjp=False)

  def test_k1_and_k8(self):
    for k in (1, 8):
      with self.subTest(k=k):
        self._check(batch=1, seq=6, k=k, num_experts=16, emb=4, use_custom_sort_vjp=True)

  def test_float32_weight_sum(self):
    self._check(batch=2, seq=4, k=2, num_experts=8, emb=8, use_custom_sort_vjp=True, float32_weight_sum=True)


class UnrouteReferenceTest(unittest.TestCase):
  """`sort_activations.unroute` / `route` (non-Mosaic path) match a NumPy reference, forward and backward."""

  def test_unroute_and_route_backward(self):
    rng = np.random.default_rng(1)
    for num_tokens, k, num_experts, emb in ((5, 1, 4, 3), (16, 4, 32, 8), (7, 8, 16, 5)):
      with self.subTest(num_tokens=num_tokens, k=k):
        selected = _routing(rng, num_tokens, k, num_experts)
        order = np.argsort(selected.reshape(-1), kind="stable")  # sorted row r holds slot order[r]
        routed = rng.standard_normal((num_tokens * k, emb)).astype(np.float32)
        inv = np.argsort(order)
        want = routed[inv].reshape(num_tokens, k, emb).sum(axis=1)

        got = sort_activations.unroute(jnp.asarray(routed), jnp.asarray(selected), False)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-5)

        # route's backward is _unroute_impl: d tokens[t] = sum over the k sorted rows holding t.
        tokens = rng.standard_normal((num_tokens, emb)).astype(np.float32)
        cot = rng.standard_normal((num_tokens * k, emb)).astype(np.float32)
        _, vjp = jax.vjp(lambda x, s=jnp.asarray(selected): sort_activations.route(x, s, False), jnp.asarray(tokens))
        (d_tokens,) = vjp(jnp.asarray(cot))
        np.testing.assert_allclose(
            np.asarray(d_tokens), cot[inv].reshape(num_tokens, k, emb).sum(axis=1), rtol=1e-5, atol=1e-5
        )


class DenseMatmulCombineDtypeTest(unittest.TestCase):
  """RoutedMoE.dense_matmul with capacity_factor > 0 keeps self.dtype."""

  def test_capacity_factor_combine_preserves_bfloat16_dtype(self):
    def _build_layer(capacity_factor):
      cfg = pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="dense_matmul_combine_dtype",
          enable_checkpointing=False,
          decoder_block="deepseek",
          dtype="bfloat16",
          weight_dtype="float32",
          sparse_matmul=False,
          megablox=False,
          float32_gate_logits=True,
          float32_weight_sum=True,
          capacity_factor=capacity_factor,
          num_experts=4,
          num_experts_per_tok=2,
          base_emb_dim=16,
          base_mlp_dim=32,
          base_moe_mlp_dim=32,
          per_device_batch_size=1,
          max_target_length=4,
      )
      devices = maxtext_utils.create_device_mesh(cfg)
      mesh = jax.sharding.Mesh(devices, cfg.mesh_axes)
      layer = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(params=0),
      )
      return cfg, mesh, layer

    def _scan_step(carry, _):
      return layer_cap(carry)[0], None

    cfg, mesh, layer_cap = _build_layer(capacity_factor=8.0)
    _, _, layer_dropless = _build_layer(capacity_factor=-1.0)
    shape = (jax.device_count(), 4, 16)
    inputs = jax.random.normal(jax.random.PRNGKey(1), shape, dtype=jnp.bfloat16)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      out_cap, _, _ = layer_cap(inputs)
      out_dropless, _, _ = layer_dropless(inputs)
      scanned, _ = jax.lax.scan(_scan_step, inputs, None, length=2)
    self.assertEqual(out_cap.dtype, jnp.bfloat16)
    self.assertEqual(scanned.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        out_cap.astype(jnp.float32),
        out_dropless.astype(jnp.float32),
        rtol=2e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()

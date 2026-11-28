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

"""CPU check of ds_ep's capped ring-of-experts dispatch buffer (StageConfig.ragged_buffer_factor).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python3 -m pytest tests/unit/experimental/mpmd/jax/ragged_buffer_test.py -q

A buffer big enough for every shard's load must give the dropless result; a small one must drop exactly what
MaxText's truncated buffer drops (`_truncate_matrix` of a shard's group sizes, ragged_sort.ring_ragged_sort's packed
path): on each EP shard, the routed (token, slot) rows in global expert order past the shard's first `rows`, rows
being MaxText's balanced load x factor, written out here rather than taken from ds_ep.  Under megablox at any
factor: a buffer that is no multiple of the row tile is padded to one (`ds_ep.gmm_row_tile`).  The reference is
numpy; MaxText's ring_ragged_unsort returns bf16 whatever the compute dtype, hence the bf16 tolerances against it.
The sizing and the group-size clamp are also held against MaxText's own functions (RoutedMoE.get_ragged_buffer_size,
_truncate_matrix).
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import testing

EP, B, S, D, F, E, K = 2, 2, 64, 32, 16, 8, 2
T = EP * B * S  # the EP group's tokens
TILE = 128  # every tm of the tests' tilings
BF16 = dict(rtol=1e-2, atol=2e-2)
BF16_GRAD = dict(rtol=2e-2, atol=6e-2)  # a gradient sums several bf16-rounded terms
# Buffers of factor x the balanced load, T // EP x K = 256 rows: 128 (a tile multiple, as the study's factors 1.5
# and 2.0 give at its scale), 179 and 332 (not one: megablox gets them padded to 256 and 384).
DROPPING = (0.5, 0.7, 1.3)


def setUpModule():
  testing.require_cpu()


def scfg(factor, gmm_impl="ragged_dot", dtype=jnp.float32):
  cfg = DeepSeekConfig(
      vocab_size=128,
      emb_dim=D,
      num_decoder_layers=1,
      first_num_dense_layers=0,
      num_query_heads=2,
      num_kv_heads=2,
      q_lora_rank=16,
      kv_lora_rank=16,
      mlp_dim=F,
      moe_mlp_dim=F,
      num_experts=E,
      num_experts_per_tok=K,
      shared_experts=1,
      routed_bias=False,
      max_target_length=S,
      dtype=dtype,
  )
  return ds_ep.StageConfig(
      model=cfg,
      fsdp=1,
      ep=EP,
      wi_tiling=(TILE,) * 9,
      wo_tiling=(TILE,) * 9,
      gmm_impl=gmm_impl,
      random_routing=False,
      rs_dtype=jnp.float32,
      ragged_buffer_factor=factor,
  )


def maxtext_rows(factor):
  """MaxText's buffer rows at `factor` for this file's EP group (get_ragged_buffer_size of the balanced size
  (T // EP) x K), written out; None for its dropless buffer."""
  return int(T // EP * K * factor) if factor > 0 else None


def inputs(seed=0, skew=True):
  r = np.random.default_rng(seed)
  # skew: experts 0-3 (EP shard 0) take most of the routing, so shard 0 overflows a small buffer.
  gate = r.normal(size=(D, E)).astype(np.float32) * 0.1
  if skew:
    gate[:, : E // 2] += 0.5
  w = {
      "gate": {"kernel": gate},
      "wi_0": r.normal(size=(E, D, F)).astype(np.float32) * 0.2,
      "wi_1": r.normal(size=(E, D, F)).astype(np.float32) * 0.2,
      "wo": r.normal(size=(E, F, D)).astype(np.float32) * 0.2,
  }
  x = r.normal(size=(EP * B, S, D)).astype(np.float32) + 0.5
  return w, x


MESH_SPECS = ({"gate": {"kernel": P()}, "wi_0": P("expert"), "wi_1": P("expert"), "wo": P("expert")}, P(ds_ep.FE))


def ring_moe(sc):
  """_ring_moe's output (not its expert counts) on an EP-wide mesh, as a function of the global weights and tokens."""
  mesh = Mesh(np.array(jax.devices()[:EP]).reshape(1, EP), ds_ep.FE)
  wspec, xspec = MESH_SPECS
  return (
      jax.shard_map(
          lambda w, x: ds_ep._ring_moe(w, x, jax.random.key(0), sc)[0],
          mesh=mesh,
          in_specs=(wspec, xspec),
          out_specs=xspec,
          check_vma=False,
      ),
      mesh,
  )


def run(sc, w, x):
  f, mesh = ring_moe(sc)
  wspec, xspec = MESH_SPECS
  put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
  return np.asarray(jax.jit(f)(jax.tree.map(put, w, wspec), put(x, xspec)))


def routing(sc, w, xf):
  """(scores, top-k ids, weights) as _ring_moe routes, in the config's dtype (bf16 scores tie and round)."""
  cfg = sc.model
  scores = ds_ep.layers.activation_fn(cfg.routed_score_func)(ds_ep.layers.dense_general(w["gate"], xf, cfg))
  _, idx = jax.lax.top_k(scores, K)
  wts = jnp.take_along_axis(scores, idx, axis=-1)
  return idx, wts / (wts.sum(-1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor


def numpy_moe(sc, w, x):
  """The capped ring's output from first principles: route, sort every (token, slot) row by expert (stable, like
  jnp.argsort), keep each EP shard's first `rows` rows of its experts' range (`maxtext_rows`), weight and sum."""
  idx, wts = (np.asarray(a, np.float64) for a in routing(sc, w, jnp.asarray(x.reshape(T, D))))
  idx = idx.astype(np.int64)
  xf = x.reshape(T, D).astype(np.float64)
  flat = idx.reshape(-1)
  order = np.argsort(flat, kind="stable")
  pos = np.empty_like(order)
  pos[order] = np.arange(order.size)  # sorted position of each (token, slot) row
  counts = np.bincount(flat, minlength=E)
  starts = np.concatenate([[0], np.cumsum(counts)])
  rows = maxtext_rows(sc.ragged_buffer_factor)
  n_local = E // EP
  shard_start = starts[(flat // n_local) * n_local]  # the first sorted row of the row's shard
  kept = np.ones_like(flat, bool) if rows is None else (pos - shard_start) < rows
  silu = lambda v: v / (1 + np.exp(-v))
  y = np.zeros_like(xf)
  for j, e in enumerate(flat):
    if kept[j]:
      t = j // K
      h = silu(xf[t] @ w["wi_0"][e]) * (xf[t] @ w["wi_1"][e])
      y[t] += wts[t, j % K] * (h @ w["wo"][e])
  return y.reshape(x.shape), kept


class Sizing(unittest.TestCase):
  """The buffer's rows and the megablox row tile: no kernel, no MaxText."""

  def test_rows(self):
    self.assertIsNone(ds_ep.ragged_buffer_rows(scfg(-1), T))
    self.assertIsNone(ds_ep.ragged_buffer_rows(scfg(EP), T))  # EP x balanced = the worst case
    for f in (1.5,) + DROPPING:
      self.assertEqual(ds_ep.ragged_buffer_rows(scfg(f), T), maxtext_rows(f))
    self.assertEqual([maxtext_rows(f) % TILE for f in DROPPING], [0, 51, 76])  # the padded cases are padded

  def test_row_tile(self):
    """The lcm of the six tm's of the two tilings (forward, dlhs, drhs each tile the rows)."""
    self.assertEqual(ds_ep.gmm_row_tile(scfg(1.0)), TILE)
    cfg = scfg(1.0).model
    for tiles, want in (("clamped", 512), ("v6e", 1024)):
      wi, wo = ds_ep.TILES[tiles](cfg)
      sc = dataclasses.replace(scfg(1.0), wi_tiling=wi, wo_tiling=wo)
      self.assertEqual(ds_ep.gmm_row_tile(sc), want, tiles)
    sc = dataclasses.replace(scfg(1.0), wi_tiling=(512,) + (128,) * 8, wo_tiling=(128,) * 3 + (384,) + (128,) * 5)
    self.assertEqual(ds_ep.gmm_row_tile(sc), 1536)

  def test_truncate_group_sizes(self):
    gs = jnp.asarray([3, 0, 5, 2], jnp.int32)
    for rows, want in ((10, [3, 0, 5, 2]), (8, [3, 0, 5, 0]), (6, [3, 0, 3, 0]), (2, [2, 0, 0, 0])):
      self.assertEqual(ds_ep.truncate_group_sizes(gs, rows).tolist(), want, rows)

  def test_needs_ragged_sort(self):
    with self.assertRaisesRegex(ValueError, "needs use_ragged_sort"):
      dataclasses.replace(scfg(2.0), use_ragged_sort=False)


@testing.needs_devices(4)
class RaggedBuffer(unittest.TestCase):

  def test_dropless_matches_numpy(self):
    w, x = inputs()
    want, _ = numpy_moe(scfg(-1), w, x)
    np.testing.assert_allclose(run(scfg(-1), w, x), want, **BF16)

  def test_roomy_buffer_is_dropless(self):
    """A buffer under the worst case but over every shard's load: the dropless output."""
    w, x = inputs(skew=False)
    sc = scfg(1.5)
    _, kept = numpy_moe(sc, w, x)
    self.assertTrue(kept.all(), "the test wants no drops at factor 1.5")
    np.testing.assert_allclose(run(sc, w, x), run(scfg(-1), w, x), rtol=1e-5, atol=1e-6)

  def test_small_buffer_drops_like_maxtext(self):
    """Each kernel at a factor whose buffer is a tile multiple and at two that are not (megablox: padded)."""
    w, x = inputs()
    dropless, _ = numpy_moe(scfg(-1), w, x)
    for impl in ("ragged_dot", "megablox"):
      for f in DROPPING:
        with self.subTest(impl=impl, factor=f):
          sc = scfg(f, impl)
          want, kept = numpy_moe(sc, w, x)
          self.assertGreater((~kept).sum(), 0, f"the test wants drops at factor {f}")
          got = run(sc, w, x)
          np.testing.assert_allclose(got, want, **BF16)
          self.assertGreater(np.abs(got - dropless).max(), 0.5, "the drops must show above the tolerance")

  def test_small_buffer_gradients_finite_and_match_reference(self):
    """d/dx and d/dw of a dropping buffer against jax.grad of the numpy-semantics reference (same kept rows), in
    bf16 compute as the runs are (MaxText's unsort returns bf16, so an fp32 primal has no matching cotangent): both
    kernels, a buffer of a tile multiple and one padded to it."""
    w, x = inputs()
    x = np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))  # the reference sees the bf16 inputs
    xb = jnp.asarray(x, jnp.bfloat16)
    probe = np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    for impl in ("ragged_dot", "megablox"):
      for factor in (0.5, 0.7):
        with self.subTest(impl=impl, factor=factor):
          sc = scfg(factor, impl, dtype=jnp.bfloat16)
          _, kept = numpy_moe(sc, w, x)
          f, _ = ring_moe(sc)
          gw, gx = jax.jit(jax.grad(lambda w, x: jnp.sum(f(w, x).astype(jnp.float32) * probe), argnums=(0, 1)))(w, xb)
          gx = gx.astype(jnp.float32)

          def ref(w, x, sc=sc, kept=kept):  # dense JAX, the kept rows of the numpy reference
            xf = x.reshape(T, D)
            idx, wts = routing(sc, w, xf)
            wts = wts.astype(jnp.float32) * jnp.asarray(kept.reshape(T, K), jnp.float32)
            h0 = jnp.einsum("td,tkdf->tkf", xf, w["wi_0"][idx])
            h1 = jnp.einsum("td,tkdf->tkf", xf, w["wi_1"][idx])
            out = jnp.einsum("tkf,tkfd->tkd", jax.nn.silu(h0) * h1, w["wo"][idx])
            return jnp.einsum("tkd,tk->td", out, wts).reshape(x.shape)

          rw, rx = jax.grad(lambda w, x: jnp.sum(ref(w, x) * probe), argnums=(0, 1))(w, x)
          self.assertTrue(all(np.isfinite(np.asarray(a, np.float32)).all() for a in jax.tree.leaves((gw, gx))))
          np.testing.assert_allclose(np.asarray(gx), np.asarray(rx), **BF16_GRAD)
          # A weight gradient sums a hundred bf16 products per element: its error scales with the leaf, not the
          # element.
          for k, a, b in [(k, gw[k], rw[k]) for k in ("wi_0", "wi_1", "wo")] + [
              ("gate", gw["gate"]["kernel"], rw["gate"]["kernel"])
          ]:
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            self.assertLess(np.abs(a - b).max(), 2e-2 * np.abs(b).max(), k)


class AgainstMaxTextSizing(unittest.TestCase):
  """ds_ep's buffer rows and group-size clamp against MaxText's own functions -- RoutedMoE.get_ragged_buffer_size
  of the balanced size, as RoutedMoE.permute calls it, and _truncate_matrix -- so a change of MaxText's sizing shows
  here, not as different buffers on the two sides of a comparison."""

  @classmethod
  def setUpClass(cls):
    from maxtext.layers import moe  # pylint: disable=import-outside-toplevel

    cls.moe = moe

  def test_buffer_rows(self):
    """For a factor > 0 MaxText sizes the buffer and passes it to the sort, which keeps every row of one as large
    as the dropless buffer (ds_ep: None); for one <= 0 it passes None."""
    get = self.moe.RoutedMoE.get_ragged_buffer_size
    base = scfg(1.0)
    for tokens, ep, experts, k in ((T, EP, E, K), (16 * 4096, 16, 576, 8), (64 * 4096, 16, 256, 8), (96, 4, 8, 3)):
      model = dataclasses.replace(base.model, num_experts=experts, num_experts_per_tok=k)
      for f in (-1.0, 0.5, 0.7, 1.0, 1.1, 1.3, 1.5, 2.0, ep, ep + 1.0):
        with self.subTest(tokens=tokens, ep=ep, experts=experts, k=k, factor=f):
          sc = dataclasses.replace(base, model=model, ep=ep, fsdp_collectives="xla", ragged_buffer_factor=f)
          want = get((tokens // ep) * k, ep, experts, k, f) if f > 0 else None
          self.assertEqual(ds_ep.ragged_buffer_rows(sc, tokens), None if want is None or want >= tokens * k else want)

  def test_group_size_clamp(self):
    r = np.random.default_rng(0)
    for _ in range(50):
      gs = jnp.asarray(r.integers(0, 40, size=int(r.integers(1, 9))), jnp.int32)
      rows = int(r.integers(1, 200))
      want = self.moe._truncate_matrix(gs[:, None], rows)[:, 0]
      np.testing.assert_array_equal(np.asarray(ds_ep.truncate_group_sizes(gs, rows)), np.asarray(want))


if __name__ == "__main__":
  unittest.main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full-buffer (dropless) ring-of-experts gather-reduces receive in-bounds indices.

With moe_routing_index_math="count", ring_ragged_sort's inverse permutation is -1 on the slots of other shards. The
full-buffer branches of `_ring_unsort_combine` and of ring_ragged_sort's backward must clamp those masked lanes before
`ragged_gather_reduce`: its SparseCore kernel reads every lane of the last row block without bounds checks.

On CPU, `ragged_gather_reduce` uses its jnp fallback, so a spy on it checks the arguments of every call instead of the
TPU fault. The tests are CPU-only: the last one deliberately passes the unclamped indices.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.ragged import ragged_gather_reduce_v2
from maxtext.kernels.ragged import ragged_sort

pytestmark = pytest.mark.cpu_only

_HIDDEN = 64
_GATHER_REDUCE_KWARGS = {
    "enforce_fallback": True,
    "flops_override": -1,
    "bytes_accessed_override": -1,
    "use_single_sparsecore": False,
}


def _bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  return bool(np.array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)))


def _routing(seed, tokens, experts, topk):
  k = jax.random.split(jax.random.PRNGKey(seed), 3)
  ids = jax.lax.top_k(jax.random.normal(k[0], (tokens, experts)), topk)[1].astype(jnp.int32)
  w = jax.random.uniform(k[1], (tokens * topk,), jnp.float32, 0.25, 0.3)
  return ids, w, k[2]


def _maps(ids, experts, ep, shard, index_math):
  """This shard's window and the inverse permutation ring_ragged_sort builds for `index_math`."""
  flat = ids.reshape(-1)
  local = experts // ep
  offs = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(jax.nn.one_hot(flat, experts, dtype=jnp.int32).sum(0))])
  start, end = offs[shard * local], offs[(shard + 1) * local]
  if index_math == "count":
    pos = ragged_sort._local_slot_positions(ids, experts, shard, ep)  # pylint: disable=protected-access
    revert = jnp.where(pos >= 0, start + pos, -1).astype(jnp.int32)
  else:
    revert = jnp.argsort(jnp.argsort(flat, stable=True)).astype(jnp.int32)
  return start, end, revert


class _Spy:
  """Wraps ragged_sort.ragged_gather_reduce; records (min index, max index, rows) of every call at run time."""

  def __init__(self, declamp=False):
    self.calls = []
    self.declamp = declamp
    self.real = ragged_sort.ragged_gather_reduce

  def __call__(self, x, indices, *args, **kwargs):
    if self.declamp:
      # The arguments before the fix: the masked lanes of a full buffer carry the raw inverse (-1 under "count").
      indices = jnp.where(kwargs["valid_rows_mask"], indices, -1)
    rows = x.shape[0]
    jax.debug.callback(lambda lo, hi: self.calls.append((int(lo), int(hi), rows)), jnp.min(indices), jnp.max(indices))
    return self.real(x, indices, *args, **kwargs)

  def out_of_bounds(self):
    return [c for c in self.calls if c[0] < 0 or c[1] >= c[2]]


_COMBINE_CASES = [(ep, m) for ep in (4, 8) for m in ("count", "sort")]


@pytest.mark.parametrize("ep,index_math", _COMBINE_CASES, ids=[f"ep{ep}-{m}" for ep, m in _COMBINE_CASES])
def test_full_buffer_combine_in_bounds_and_unchanged(ep, index_math):
  tokens, experts, topk = 256, 2 * ep, 4
  n = tokens * topk
  for seed in range(3):
    ids, w, key = _routing(seed, tokens, experts, topk)
    x = jax.random.normal(key, (n, _HIDDEN), jnp.float32).astype(jnp.bfloat16)  # full buffer: rows at global positions
    for shard in range(ep):
      start, end, revert = _maps(ids, experts, ep, shard, index_math)
      if index_math == "count":
        assert int(jnp.sum(revert < 0)) > 0, "count must leave other shards' slots at -1"
      spy = _Spy()
      with mock.patch.object(ragged_sort, "ragged_gather_reduce", spy):
        out = jax.jit(
            lambda x, r, w, s, e: ragged_sort._ring_unsort_combine(  # pylint: disable=protected-access
                x, r, w, s, e, topk, _GATHER_REDUCE_KWARGS
            )
        )(x, revert, w, start, end)
        out.block_until_ready()
      jax.effects_barrier()
      assert len(spy.calls) == 1, spy.calls
      assert not spy.out_of_bounds(), f"seed{seed} shard{shard}: indices out of [0, {n}): {spy.calls}"
      # The call before the fix: the raw inverse. The masked lanes are discarded, so the output is the same bits.
      valid = (revert >= start) & (revert < end)
      before = ragged_gather_reduce_v2.ragged_gather_reduce(
          x=x,
          indices=revert,
          topk_weights=w,
          valid_rows_mask=valid,
          reduce_group_size=topk,
          enforce_fallback=True,
      )
      assert _bits_equal(out, before), f"seed{seed} shard{shard}: output changed"
      # float64 reference: each token's routing-weighted sum over its slots in this shard's window.
      x_np, revert_np, w_np, valid_np = (np.asarray(a) for a in (x.astype(jnp.float32), revert, w, valid))
      gathered = x_np[np.where(valid_np, revert_np, 0)] * w_np[:, None]
      ref = np.where(valid_np[:, None], gathered, 0.0).reshape((tokens, topk, _HIDDEN)).sum(1)
      np.testing.assert_allclose(np.asarray(out, np.float32), ref, rtol=2e-2, atol=2e-2)


def _op(ep, index_math, tokens=128, topk=4, seed=0):
  """ring_ragged_sort (full buffer) -> ring_ragged_unsort per shard, vmapped over "ep": (out, dx, dw)."""
  experts = 2 * ep
  local = experts // ep
  ids, w, key = _routing(seed, tokens, experts, topk)
  k = jax.random.split(key, 2)
  x = jax.random.normal(k[0], (tokens, _HIDDEN), jnp.float32).astype(jnp.bfloat16)
  cot = jax.random.normal(k[1], (ep, tokens, _HIDDEN), jnp.float32).astype(jnp.bfloat16)

  def per_shard(x, w, cot):
    def f(x, w):
      sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
          x, ids, experts, topk, "ep", ep, buffer_size=None, index_math=index_math
      )
      assert sorted_x.shape[0] == tokens * topk  # the full buffer
      return ragged_sort.ring_ragged_unsort(
          sorted_x * 2, gs, revert, topk, local, "ep", w, topk_argsort_indices=perm, index_math=index_math
      )

    out, vjp = jax.vjp(f, x, w)
    dx, dw = vjp(cot)
    return out, dx, dw

  return jax.jit(jax.vmap(per_shard, in_axes=(None, None, 0), axis_name="ep"))(x, w, cot)


@pytest.mark.parametrize("ep", [4, 8])
def test_full_buffer_op_in_bounds_and_count_matches_sort(ep):
  results = {}
  for index_math in ("count", "sort"):
    spy = _Spy()
    with mock.patch.object(ragged_sort, "ragged_gather_reduce", spy):
      results[index_math] = jax.block_until_ready(_op(ep, index_math))
    jax.effects_barrier()
    # Per shard: one combine (forward) and one sort backward.
    assert len(spy.calls) == 2 * ep, spy.calls
    assert not spy.out_of_bounds(), f"{index_math}: indices out of bounds: {spy.out_of_bounds()}"
  for name, a, b in zip(("out", "dx", "dw"), results["count"], results["sort"]):
    assert _bits_equal(a, b), f"ep{ep}: {name} differs between count and sort"


def test_unclamped_indices_are_flagged():
  """The arguments before the fix (-1 on the masked lanes) fail the bounds check, in both call sites."""
  ep = 4
  spy = _Spy(declamp=True)
  with mock.patch.object(ragged_sort, "ragged_gather_reduce", spy):
    jax.block_until_ready(_op(ep, "count"))
  jax.effects_barrier()
  assert len(spy.calls) == 2 * ep, spy.calls
  assert len(spy.out_of_bounds()) == 2 * ep, spy.calls

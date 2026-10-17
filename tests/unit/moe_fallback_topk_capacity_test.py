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
"""moe_fallback_topk_capacity against the key off, on CPU.

Under router replay (forced routing) RoutedMoE.get_topk computes the router's own top-k ids for every token and keeps
them only for tokens without valid forced ids (and, under moe_exclude_padding_from_routing, never for padding). With
moe_fallback_topk_capacity = C > 0, `moe.fallback_topk_indices_compacted` computes them only for the tokens that use
them: the first C such rows are gathered in token order, top-k'd and written back; a lax.cond runs the top-k on every
row instead when more than C tokens use them. top-k is row-wise, so get_topk's weights, ids and gradients must be
bit-identical to C = 0.

  - The key: an int, default 0 in types.py and base.yml, resolved by pyconfig; a negative value is refused.
  - The helper directly, for both top-k functions get_topk uses (argmax passes for moe_routing_index_math="count",
    lax.top_k for "sort"), random / exactly tied / duplicated score rows, and using-token counts on both sides of the
    capacity (0, 1, C - 1, C, C + 1, all): every using token gets exactly the full top-k's ids; on the compacted
    branch every other token gets 0; above the capacity every token gets the full top-k's ids.
  - get_topk, the Qwen3.5 forced-routing path (norm_topk_prob), forward and VJP: weights, ids and d(gate_logits) with
    C in {1, 7, 64, using - 1, using} bit-identical to C = 0, for {"count", "sort"} x exclude_padding {on, off}, on
    forced ids mixing valid tokens, unset (-1) slots, duplicate ids and padding rows.
  - Structure, two-sided: get_topk's program contains a lax.cond with C > 0 and none with C = 0, so a key that never
    reaches get_topk cannot pass.
  - Deliberately wrong helpers (a write-back one row off, no lax.cond, the cond one row too permissive) must be
    caught by the same get_topk comparison, and the real helper must not be.
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maxtext.common import common_types as ctypes
from maxtext.configs import types as mt_types
from maxtext.layers import moe

_KEY = "moe_fallback_topk_capacity"
# The key this file tests. On a MaxText without it this line raises, so the file cannot be collected there.
_KEY_FIELD = mt_types.MoEGeneral.model_fields[_KEY]
_TOPK_FNS = {"count": moe.topk_indices_by_argmax, "sort": moe._top_k_ids}  # pylint: disable=protected-access


def _bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  return bool(np.array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)))


def _pyconfig(**overrides):
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  common = {
      "run_name": "fallback_topk_capacity_key",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **common, **overrides)


def test_key_is_an_int_default_zero():
  assert _KEY_FIELD.annotation is int
  assert _KEY_FIELD.default == 0
  base_yml = os.path.join(os.path.dirname(mt_types.__file__), "base.yml")
  with open(base_yml, encoding="utf-8") as f:
    assert yaml.safe_load(f)[_KEY] == 0
  assert getattr(_pyconfig(), _KEY) == 0
  assert getattr(_pyconfig(**{_KEY: 128}), _KEY) == 128


def test_negative_capacity_is_refused():
  with pytest.raises(ValueError, match=_KEY):
    _pyconfig(**{_KEY: -1})


# ------------------------------------------------------------------------------------------------- the helper, directly
def _scores(kind, n, e, seed):
  logits = jax.random.normal(jax.random.PRNGKey(seed), (1, n, e), jnp.float32) * 2.0
  if kind == "ties":  # quantized logits: many exactly equal probabilities in each row
    logits = jnp.round(logits * 2.0) / 2.0
  elif kind == "duprows":  # every 4th row repeats the row before it
    logits = logits.at[:, 1::4].set(logits[:, 0::4][:, : logits[:, 1::4].shape[1]])
  return jax.nn.softmax(logits, axis=-1)


def _uses(n, count, seed):
  rng = np.random.default_rng(seed)
  mask = np.zeros((1, n), bool)
  if count:
    mask[0, rng.choice(n, size=count, replace=False)] = True
  return jnp.asarray(mask)


_HELPER_SHAPES = ((64, 16, 4, 8), (1000, 64, 8, 32), (4096, 512, 10, 128))  # (tokens, experts, k, capacity)


@pytest.mark.parametrize("index_math", sorted(_TOPK_FNS))
@pytest.mark.parametrize("kind", ("random", "ties", "duprows"))
def test_helper_gives_every_using_token_the_full_top_k(index_math, kind):
  for n, e, k, cap in _HELPER_SHAPES:
    topk_fn = functools.partial(_TOPK_FNS[index_math], k=k)
    full = jax.jit(topk_fn)
    helper = jax.jit(lambda s, u, f=topk_fn, k=k, cap=cap: moe.fallback_topk_indices_compacted(s, u, k, cap, f))
    for count in sorted({0, 1, cap - 1, cap, cap + 1, n}):
      scores = _scores(kind, n, e, seed=n + count + len(kind))
      uses = _uses(n, count, seed=7 * n + count)
      want = np.asarray(full(scores))
      got = np.asarray(helper(scores, uses))
      u = np.asarray(uses)
      case = f"n={n} e={e} k={k} cap={cap} using={count} {kind} {index_math}"
      assert got.dtype == np.int32, case
      assert _bits_equal(got[u], want[u].astype(np.int32)), case
      if count <= cap:  # the compacted branch: a token that does not use the ids gets 0 (never read by get_topk)
        assert not got[~u].any(), case
      else:  # the lax.cond's every-row branch: identical to the full top-k for every token
        assert _bits_equal(got, want.astype(np.int32)), case


# ------------------------------------------------------------------------------------------------------ get_topk
class _Cfg:
  """The MaxTextConfig attributes RoutedMoE.get_topk reads on the Qwen3.5 forced-routing path."""

  def __init__(self, capacity, index_math, exclude_padding):
    self.model_name = "qwen3.5-397b-a17b"
    self.decoder_block = ctypes.DecoderBlockType.QWEN3_5
    self.norm_topk_prob = True
    self.use_random_routing = False
    self.moe_routing_index_math = index_math
    self.moe_exclude_padding_from_routing = exclude_padding
    self.moe_fallback_topk_capacity = capacity
    self.float32_weight_sum = True
    self.model_call_mode = "train"
    self.fuse_expert_scales = False


class _Moe:
  """Host for calling RoutedMoE.get_topk unbound (as router_replay_test.py's DummyRoutedMoE does)."""

  def __init__(self, cfg, num_experts, k):
    self.config = cfg
    self.num_experts = num_experts
    self.num_experts_per_tok = k
    self.dtype = jnp.bfloat16
    self.is_hash_routing = False
    self.per_expert_scale = None


def _forced_ids(n, e, k, n_unset, n_dup, pad_frac, seed):
  """Valid unique ids for most tokens, n_unset tokens with one -1 slot, n_dup tokens with a repeated id, and a
  pad_frac of the tokens folded to padding rows by moe.fold_padding_into_forced_routing."""
  rng = np.random.default_rng(seed)
  ids = np.stack([rng.choice(e, size=k, replace=False) for _ in range(n)]).astype(np.int32)[None]
  order = rng.permutation(n)
  unset, dup = order[:n_unset], order[n_unset : n_unset + n_dup]
  ids[0, unset, rng.integers(0, k, size=n_unset)] = -1
  ids[0, dup, 1] = ids[0, dup, 0]
  token_mask = np.ones((1, n), np.int32)
  token_mask[0, rng.random(n) < pad_frac] = 0
  return moe.fold_padding_into_forced_routing(jnp.asarray(ids), jnp.asarray(token_mask), k)


def _num_using(forced, e, exclude_padding):
  slot_valid = (forced >= 0) & (forced < e)
  s = jnp.sort(forced, axis=-1)
  dup = jnp.any(s[..., 1:] == s[..., :-1], axis=-1)
  uses = ~(jnp.all(slot_valid, axis=-1) & ~dup)
  if exclude_padding:
    uses &= ~jnp.all(forced == moe.PAD_TOKEN_ROUTED_EXPERT, axis=-1)
  return int(jnp.sum(uses))


def _get_topk(cfg, e, k, logits, forced):
  return moe.RoutedMoE.get_topk(_Moe(cfg, e, k), logits, logits, forced_routed_experts=forced)


def _fwd_vjp(cfg, e, k, logits, forced, g):
  """get_topk's (weights, ids) and d(gate_logits) of sum(weights * g), each jitted."""
  w, ids = jax.jit(lambda lg: _get_topk(cfg, e, k, lg, forced))(logits)

  def loss(lg):
    ww, _ = _get_topk(cfg, e, k, lg, forced)
    return jnp.sum(ww.astype(jnp.float32) * g)

  return w, ids, jax.jit(jax.grad(loss))(logits)


# (tokens, experts, k, unset, duplicate, padding fraction): below the capacities 64 / using, and above all but `using`.
_TOPK_CASES = ((512, 64, 8, 10, 5, 0.1), (512, 64, 8, 100, 60, 0.0))


def _topk_mismatches(index_math, exclude_padding):
  """[(case, capacity)] where get_topk with the capacity differs from capacity 0 in weights, ids or gradient."""
  bad = []
  for n, e, k, n_unset, n_dup, pad in _TOPK_CASES:
    forced = _forced_ids(n, e, k, n_unset, n_dup, pad, seed=n + n_unset)
    logits = jax.random.normal(jax.random.PRNGKey(n + n_dup), (1, n, e), jnp.float32) * 2.0
    g = jax.random.normal(jax.random.PRNGKey(3 + n_unset), (1, n, k), jnp.float32)
    base = _fwd_vjp(_Cfg(0, index_math, exclude_padding), e, k, logits, forced, g)
    using = _num_using(forced, e, exclude_padding)
    caps = {1, 7, 64}
    if using >= 2:  # the cond's boundary: exactly full (compacted) and one over (every row)
      caps |= {using - 1, using}
    for cap in sorted(caps):
      got = _fwd_vjp(_Cfg(cap, index_math, exclude_padding), e, k, logits, forced, g)
      if not all(_bits_equal(a, b) for a, b in zip(got, base)):
        bad.append((f"n={n} using={using} {index_math} exclude_padding={exclude_padding}", cap))
  return bad


@pytest.mark.parametrize("index_math", ("count", "sort"))
@pytest.mark.parametrize("exclude_padding", (True, False))
def test_get_topk_forward_and_vjp_are_bit_identical(index_math, exclude_padding):
  assert not _topk_mismatches(index_math, exclude_padding)


def _has_cond(capacity):
  n, e, k = 128, 16, 4
  forced = _forced_ids(n, e, k, 5, 3, 0.1, seed=1)
  logits = jax.random.normal(jax.random.PRNGKey(2), (1, n, e), jnp.float32)
  jaxpr = jax.make_jaxpr(lambda lg: _get_topk(_Cfg(capacity, "count", True), e, k, lg, forced))(logits)
  return "cond[" in str(jaxpr)


def test_capacity_selects_the_compacted_program():
  assert not _has_cond(0)
  assert _has_cond(64)


# --------------------------------------------------------------------------------- deliberately wrong helpers
def _broken_helper(kind):
  """fallback_topk_indices_compacted with one deliberate defect."""

  def helper(scores, uses, k, capacity, topk_fn):
    lead = scores.shape[:-1]
    n = int(np.prod(lead))
    cap = min(int(capacity), n)
    flat = scores.reshape(n, scores.shape[-1])
    fu = uses.reshape(n)
    rank = jnp.cumsum(fu.astype(jnp.int32)) - 1
    slot = jnp.where(fu, rank, -1)
    num_using = rank[-1] + 1

    def compacted(ops):
      flat, slot = ops
      ri = jax.lax.broadcasted_iota(jnp.int32, (cap, n), 0)
      ti = jax.lax.broadcasted_iota(jnp.int32, (cap, n), 1)
      tok = jnp.sum(jnp.where(slot[None, :] == ri, ti, 0), axis=1)
      ids = topk_fn(jnp.take(flat, tok, axis=0)).astype(jnp.int32)
      if kind == "row_off":  # row c's ids land on row c + 1's token
        ids = jnp.roll(ids, 1, axis=0)
      hit = slot[None, :] == ri
      return jnp.sum(jnp.where(hit[None, :, :], ids.T[:, :, None], jnp.zeros((), jnp.int32)), axis=1).T

    def every_row(ops):
      return topk_fn(ops[0]).astype(jnp.int32)

    if kind == "no_cond":  # the compacted branch even above the capacity
      out = compacted((flat, slot))
    elif kind == "cond_off_by_one":  # num_using == cap + 1 takes the compacted branch and loses its last row
      out = jax.lax.cond(num_using <= cap + 1, compacted, every_row, (flat, slot))
    else:
      out = jax.lax.cond(num_using <= cap, compacted, every_row, (flat, slot))
    return out.reshape(lead + (k,))

  return helper


@pytest.mark.parametrize("kind", ("row_off", "no_cond", "cond_off_by_one", "correct"))
def test_wrong_helpers_are_caught(kind, monkeypatch):
  monkeypatch.setattr(moe, "fallback_topk_indices_compacted", _broken_helper(kind))
  mismatches = _topk_mismatches("count", True)
  if kind == "correct":  # the control: a correct copy through the same harness is never caught
    assert not mismatches
  else:
    assert mismatches, kind

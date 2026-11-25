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
"""Unit tests for `moe_fallback_topk_capacity` and `fallback_topk_indices_compacted`."""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.configs import types as mt_types
from maxtext.layers import moe
from maxtext.utils import globals as mt_globals
from tests.utils.test_helpers import get_test_config_path

_KEY = "moe_fallback_topk_capacity"
_KEY_FIELD = mt_types.MoEGeneral.model_fields[_KEY]
_TOPK_FNS = {"count": moe.topk_indices_by_argmax, "sort": moe._top_k_ids}  # pylint: disable=protected-access


def _bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  return bool(np.array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)))


def _pyconfig(**overrides):
  common = {
      "run_name": "fallback_topk_capacity_key",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **common, **overrides)


def test_key_is_tier2_int_default_zero_and_validates_edges():
  assert _KEY_FIELD.annotation is int
  assert _KEY_FIELD.default == 0
  base_yml = os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
  with open(base_yml, encoding="utf-8") as f:
    assert _KEY not in yaml.safe_load(f)
  assert getattr(_pyconfig(), _KEY) == 0
  assert getattr(_pyconfig(**{_KEY: 128}), _KEY) == 128

  with pytest.raises(ValueError, match=_KEY):
    _pyconfig(**{_KEY: -1})
  with pytest.raises(ValueError, match="te_moe_block=False"):
    _pyconfig(**{_KEY: 16, "te_moe_block": True, "sparse_matmul": True})
  with pytest.raises(ValueError, match="use_random_routing=False"):
    _pyconfig(**{_KEY: 16, "use_random_routing": True, "ici_expert_parallelism": 2})


def test_helper_empty_tokens_edge_case():
  scores = jnp.zeros((2, 0, 16), dtype=jnp.float32)
  uses = jnp.zeros((2, 0), dtype=jnp.bool_)
  out = moe.fallback_topk_indices_compacted(
      scores,
      uses,
      4,
      8,
      functools.partial(moe.topk_indices_by_argmax, k=4),
  )
  assert out.shape == (2, 0, 4)
  assert out.dtype == jnp.int32


def _scores(kind, n, e, seed):
  logits = jax.random.normal(jax.random.PRNGKey(seed), (1, n, e), jnp.float32) * 2.0
  if kind == "ties":
    logits = jnp.round(logits * 2.0) / 2.0
  elif kind == "duprows":
    logits = logits.at[:, 1::4].set(logits[:, 0::4][:, : logits[:, 1::4].shape[1]])
  return logits


def _uses(n, count, seed):
  rng = np.random.default_rng(seed)
  mask = np.zeros((1, n), bool)
  if count:
    mask[0, rng.choice(n, size=count, replace=False)] = True
  return jnp.asarray(mask)


_HELPER_SHAPES = ((64, 16, 4, 8), (512, 64, 8, 32), (1024, 128, 8, 64))


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
      if count <= cap:
        assert not got[~u].any(), case
      else:
        assert _bits_equal(got, want.astype(np.int32)), case


class _Cfg:
  """Minimal config attributes RoutedMoE.get_topk reads."""

  def __init__(
      self,
      capacity,
      index_math,
      decoder_block=ctypes.DecoderBlockType.QWEN3_5,
      dense_vjp=False,
      model_name="qwen3.5-397b-a17b",
      n_routing_groups=-1,
      topk_routing_group=-1,
      routed_score_func="",
      routed_scaling_factor=1.0,
  ):
    self.model_name = model_name
    self.decoder_block = decoder_block
    self.norm_topk_prob = decoder_block == ctypes.DecoderBlockType.QWEN3_5
    self.use_random_routing = False
    self.te_moe_block = False
    self.moe_routing_index_math = index_math
    self.moe_fallback_topk_capacity = capacity
    self.router_topk_matmul_vjp = dense_vjp
    self.routed_score_func = routed_score_func
    self.routed_scaling_factor = routed_scaling_factor
    self.n_routing_groups = n_routing_groups
    self.topk_routing_group = topk_routing_group
    self.moe_routing_maps = "remat"
    self.model_call_mode = "train"
    self.fuse_expert_scales = False


class _Moe:
  """Minimal RoutedMoE host for direct get_topk verification."""

  expert_group_mask = moe.RoutedMoE.expert_group_mask
  deepseek_routing = moe.RoutedMoE.deepseek_routing
  deepseek_scale_weights = moe.RoutedMoE.deepseek_scale_weights

  def __init__(self, cfg, num_experts, k, is_hash_routing=False, tid2eid=None):
    self.config = cfg
    self.num_experts = num_experts
    self.num_experts_per_tok = k
    self.dtype = jnp.float32
    self.is_hash_routing = is_hash_routing
    self.tid2eid = tid2eid
    self.per_expert_scale = None


def _forced_and_mask(n, e, k, n_unset, n_dup, pad_frac, seed):
  rng = np.random.default_rng(seed)
  ids = np.stack([rng.choice(e, size=k, replace=False) for _ in range(n)]).astype(np.int32)[None]
  order = rng.permutation(n)
  unset, dup = order[:n_unset], order[n_unset : n_unset + n_dup]
  ids[0, unset, rng.integers(0, k, size=n_unset)] = -1
  ids[0, dup, 1] = ids[0, dup, 0]
  token_mask = rng.random((1, n)) >= pad_frac
  return jnp.asarray(ids), jnp.asarray(token_mask)


def _num_using(forced, e, token_mask):
  slot_valid = (forced >= 0) & (forced < e)
  s = jnp.sort(forced, axis=-1)
  dup = jnp.any(s[..., 1:] == s[..., :-1], axis=-1)
  uses = ~(jnp.all(slot_valid, axis=-1) & ~dup)
  if token_mask is not None:
    uses &= token_mask
  return int(jnp.sum(uses))


def _get_topk(cfg, e, k, logits, forced=None, token_mask=None):
  return moe.RoutedMoE.get_topk(
      _Moe(cfg, e, k),
      logits,
      logits,
      forced_routed_experts=forced,
      token_mask=token_mask,
  )


def _fwd_vjp(cfg, e, k, logits, forced, token_mask, g):
  w, ids = jax.jit(lambda lg: _get_topk(cfg, e, k, lg, forced, token_mask))(logits)

  def loss(lg):
    ww, _ = _get_topk(cfg, e, k, lg, forced, token_mask)
    return jnp.sum(ww.astype(jnp.float32) * g)

  return w, ids, jax.jit(jax.grad(loss))(logits)


_TOPK_CASES = ((256, 32, 4, 8, 4, 0.15), (256, 32, 4, 50, 30, 0.0))


def _topk_mismatches(index_math, use_token_mask, decoder_block=ctypes.DecoderBlockType.QWEN3_5):
  """Returns (case, capacity) pairs whose compacted get_topk/VJP differs from full top-k."""
  bad = []
  for n, e, k, n_unset, n_dup, pad in _TOPK_CASES:
    forced, mask = _forced_and_mask(n, e, k, n_unset, n_dup, pad if use_token_mask else 0.0, seed=n + n_unset)
    tmask = mask if use_token_mask else None
    logits = jax.random.normal(jax.random.PRNGKey(n + n_dup), (1, n, e), jnp.float32) * 2.0
    g = jax.random.normal(jax.random.PRNGKey(3 + n_unset), (1, n, k), jnp.float32)
    # Reference: capacity = n evaluates every row unconditionally via the overflow branch / full top-k.
    ref = _fwd_vjp(_Cfg(n, index_math, decoder_block), e, k, logits, forced, tmask, g)

    # Also verify forced tokens match uncompacted forced get_topk and fallback tokens match unforced get_topk!
    unforced_w, unforced_ids, _ = _fwd_vjp(_Cfg(0, index_math, decoder_block), e, k, logits, None, tmask, g)
    slot_valid = (np.asarray(forced) >= 0) & (np.asarray(forced) < e)
    sorted_f = np.sort(np.asarray(forced), axis=-1)
    has_dup = np.any(sorted_f[..., 1:] == sorted_f[..., :-1], axis=-1)
    is_forced = np.all(slot_valid, axis=-1) & ~has_dup
    is_pad = ~np.asarray(tmask) if tmask is not None else np.zeros_like(is_forced)
    is_fallback = ~is_forced & ~is_pad
    if not (
        _bits_equal(np.asarray(ref[1])[is_fallback], np.asarray(unforced_ids)[is_fallback])
        and np.allclose(np.asarray(ref[0])[is_fallback], np.asarray(unforced_w)[is_fallback], rtol=1e-6, atol=1e-6)
    ):
      bad.append((f"n={n} ref_vs_unforced {index_math} mask={use_token_mask} {decoder_block}", n))
      continue
    if np.any(is_pad):
      assert np.all(np.asarray(ref[1])[is_pad] == -1)
      assert np.all(np.asarray(ref[0])[is_pad] == 0.0)
      assert np.all(np.asarray(ref[2])[is_pad] == 0.0)

    using = _num_using(forced, e, tmask)
    caps = {1, 7, 32}
    if using >= 2:
      caps |= {using - 1, using, using + 1}
    for cap in sorted(caps):
      got = _fwd_vjp(_Cfg(cap, index_math, decoder_block), e, k, logits, forced, tmask, g)
      if not all(_bits_equal(a, b) for a, b in zip(got, ref)):
        bad.append((f"n={n} using={using} {index_math} mask={use_token_mask} {decoder_block}", cap))
  return bad


@pytest.mark.parametrize("index_math", ("count", "sort"))
@pytest.mark.parametrize("use_token_mask", (True, False))
@pytest.mark.parametrize(
    "decoder_block",
    (ctypes.DecoderBlockType.QWEN3_5, ctypes.DecoderBlockType.MIXTRAL, ctypes.DecoderBlockType.GEMMA4),
)
def test_get_topk_forward_and_vjp_are_bit_identical(index_math, use_token_mask, decoder_block):
  assert not _topk_mismatches(index_math, use_token_mask, decoder_block)


def test_padding_tokens_do_not_consume_fallback_capacity():
  """With 90% padding and 5 fallback tokens, C=8 must stay on the compacted branch instead of overflowing."""
  n, e, k = 128, 16, 4
  forced = jnp.broadcast_to(jnp.array([0, 1, 2, 3], jnp.int32), (1, n, k))
  # 5 real fallback tokens (-1), 23 real forced tokens, 100 padding tokens (-1 with token_mask=False).
  forced = forced.at[:, :5, :].set(-1).at[:, 28:, :].set(-1)
  token_mask = jnp.arange(n)[None, :] < 28
  logits = jax.random.normal(jax.random.PRNGKey(9), (1, n, e), jnp.float32)

  cfg = _Cfg(8, "count")
  w_c8, ids_c8 = _get_topk(cfg, e, k, logits, forced, token_mask)
  w_all, ids_all = _get_topk(_Cfg(n, "count"), e, k, logits, forced, token_mask)
  assert _bits_equal(w_c8, w_all)
  assert _bits_equal(ids_c8, ids_all)

  # Verify jaxpr has cond only when capacity > 0 (zero default JAXpr drift when capacity == 0).
  jaxpr_0 = str(jax.make_jaxpr(lambda lg: _get_topk(_Cfg(0, "count"), e, k, lg, forced, token_mask))(logits))
  jaxpr_8 = str(jax.make_jaxpr(lambda lg: _get_topk(_Cfg(8, "count"), e, k, lg, forced, token_mask))(logits))
  assert "cond[" not in jaxpr_0
  assert "cond[" in jaxpr_8


def _broken_helper(kind):
  """Deliberately defective fallback_topk_indices_compacted for two-sided test validation."""

  def helper(scores, uses, k, capacity, topk_fn):
    lead = scores.shape[:-1]
    n = int(np.prod(lead))
    cap = min(int(capacity), n)
    flat = scores.reshape(n, scores.shape[-1])
    fu = uses.reshape(n).astype(jnp.bool_)
    rank = jnp.cumsum(fu.astype(jnp.int32)) - 1
    slot = jnp.where(fu, rank, -1)
    num_using = jnp.sum(fu, dtype=jnp.int32)

    def compacted(ops):
      flat, slot = ops
      ri = jax.lax.broadcasted_iota(jnp.int32, (cap, n), 0)
      ti = jax.lax.broadcasted_iota(jnp.int32, (cap, n), 1)
      tok = jnp.sum(jnp.where(slot[None, :] == ri, ti, 0), axis=1)
      ids = topk_fn(jnp.take(flat, tok, axis=0)).astype(jnp.int32)
      if kind == "row_off":
        ids = jnp.roll(ids, 1, axis=0)
      hit = slot[None, :] == ri
      return jnp.sum(jnp.where(hit[None, :, :], ids.T[:, :, None], jnp.zeros((), jnp.int32)), axis=1).T

    def every_row(ops):
      return topk_fn(ops[0]).astype(jnp.int32)

    if kind == "no_cond":
      out = compacted((flat, slot))
    elif kind == "cond_off_by_one":
      out = jax.lax.cond(num_using <= cap + 1, compacted, every_row, (flat, slot))
    else:
      out = jax.lax.cond(num_using <= cap, compacted, every_row, (flat, slot))
    return out.reshape(lead + (k,))

  return helper


@pytest.mark.parametrize("kind", ("row_off", "no_cond", "cond_off_by_one", "correct"))
def test_wrong_helpers_are_caught(kind, monkeypatch):
  monkeypatch.setattr(moe, "fallback_topk_indices_compacted", _broken_helper(kind))
  mismatches = _topk_mismatches("count", True)
  if kind == "correct":
    assert not mismatches
  else:
    assert mismatches, kind


@pytest.mark.parametrize("index_math", ("count", "sort"))
@pytest.mark.parametrize("dense_vjp", (False, True))
def test_deepseek_group_mask_and_pre_bias_logits_fallback_routing(index_math, dense_vjp):
  """Fallback rows on DeepSeek apply expert_group_mask, gather pre_bias_logits, and leak zero grad to gate.bias."""
  n, e, k, n_groups, topk_groups = 64, 32, 4, 8, 2
  raw_logits = jax.random.normal(jax.random.PRNGKey(41), (1, n, e), jnp.float32) * 1.5
  pre_bias = jax.nn.sigmoid(raw_logits)
  # Non-trivial per-expert routed bias large enough to steer group selection and corrupt weights if used directly.
  bias = jax.random.normal(jax.random.PRNGKey(42), (e,), jnp.float32) * 0.4
  forced, tmask = _forced_and_mask(n, e, k, n_unset=6, n_dup=4, pad_frac=0.15, seed=77)
  cot = jax.random.normal(jax.random.PRNGKey(43), (1, n, k), jnp.float32)

  def make_ds_cfg(cap):
    return _Cfg(
        cap,
        index_math,
        decoder_block=ctypes.DecoderBlockType.DEEPSEEK,
        dense_vjp=dense_vjp,
        model_name="deepseek3-671b",
        n_routing_groups=n_groups,
        topk_routing_group=topk_groups,
        routed_score_func="sigmoid",
        routed_scaling_factor=2.5,
    )

  # Unforced reference (standard deepseek_routing + deepseek_scale_weights).
  unforced_host = _Moe(make_ds_cfg(0), e, k)
  unforced_w, unforced_ids = moe.RoutedMoE.get_topk(
      unforced_host, pre_bias + bias, pre_bias, forced_routed_experts=None, token_mask=tmask
  )
  # Standard forced-only reference (capacity=0) for token_is_forced rows.
  forced0_w, forced0_ids = moe.RoutedMoE.get_topk(
      unforced_host, pre_bias + bias, pre_bias, forced_routed_experts=forced, token_mask=tmask
  )

  slot_valid = (np.asarray(forced) >= 0) & (np.asarray(forced) < e)
  sorted_f = np.sort(np.asarray(forced), axis=-1)
  is_forced = np.all(slot_valid, axis=-1) & ~np.any(sorted_f[..., 1:] == sorted_f[..., :-1], axis=-1)
  is_pad = ~np.asarray(tmask)
  is_fallback = ~is_forced & ~is_pad
  all_fallback_forced = jnp.full((1, n, k), -1, dtype=jnp.int32)

  for cap in (8, 16, n):
    host = _Moe(make_ds_cfg(cap), e, k)
    got_w, got_ids = jax.jit(
        lambda pb, b, h=host: moe.RoutedMoE.get_topk(h, pb + b, pb, forced_routed_experts=forced, token_mask=tmask)
    )(pre_bias, bias)
    # Fallback rows match unforced DeepSeek group-masked + pre_bias_logits routing bit-for-bit.
    assert _bits_equal(np.asarray(got_ids)[is_fallback], np.asarray(unforced_ids)[is_fallback]), cap
    np.testing.assert_allclose(np.asarray(got_w)[is_fallback], np.asarray(unforced_w)[is_fallback], rtol=1e-6, atol=1e-6)
    # Every fallback row's experts come from at most topk_groups distinct groups.
    experts_per_group = e // n_groups
    for row_ids in np.asarray(got_ids)[is_fallback]:
      assert len(set((row_ids // experts_per_group).tolist())) <= topk_groups
    # Forced non-pad rows preserve origin/main forced routing behavior.
    forced_real = is_forced & ~is_pad
    assert _bits_equal(np.asarray(got_ids)[forced_real], np.asarray(forced0_ids)[forced_real]), cap
    assert _bits_equal(np.asarray(got_w)[forced_real], np.asarray(forced0_w)[forced_real]), cap

    # Gradient wrt gate.bias from fallback rows must be identically zero.
    grad_pb, grad_bias = jax.jit(
        jax.grad(
            lambda pb, b, h=host: jnp.sum(
                moe.RoutedMoE.get_topk(h, pb + b, pb, forced_routed_experts=all_fallback_forced, token_mask=tmask)[0]
                * cot
            ),
            argnums=(0, 1),
        )
    )(pre_bias, bias)
    np.testing.assert_array_equal(np.asarray(grad_bias), 0.0)
    assert float(np.max(np.abs(np.asarray(grad_pb)))) > 0.0


def test_hash_routed_fallback_uses_tid2eid():
  """Hash-routed MoE layers resolve fallback rows via tid2eid[input_ids] and pre_bias_logits."""
  n, e, k, vocab = 32, 16, 4, 128
  tid2eid = moe.Tid2EidVar(moe.init_tid2eid(jax.random.PRNGKey(51), vocab, e, k))
  input_ids = jax.random.randint(jax.random.PRNGKey(52), (1, n), 0, vocab, dtype=jnp.int32)
  pre_bias = jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(53), (1, n, e), jnp.float32))
  forced, tmask = _forced_and_mask(n, e, k, n_unset=5, n_dup=3, pad_frac=0.1, seed=88)

  cfg_unforced = _Cfg(
      0,
      "sort",
      decoder_block=ctypes.DecoderBlockType.DEEPSEEK4,
      model_name="deepseek4-test",
      routed_score_func="sigmoid",
      routed_scaling_factor=2.5,
  )
  unforced_host = _Moe(cfg_unforced, e, k, is_hash_routing=True, tid2eid=tid2eid)
  unforced_w, unforced_ids = moe.RoutedMoE.get_topk(
      unforced_host, pre_bias, pre_bias, input_ids=input_ids, forced_routed_experts=None, token_mask=tmask
  )

  cfg_fb = _Cfg(
      8,
      "count",
      decoder_block=ctypes.DecoderBlockType.DEEPSEEK4,
      model_name="deepseek4-test",
      routed_score_func="sigmoid",
      routed_scaling_factor=2.5,
  )
  fb_host = _Moe(cfg_fb, e, k, is_hash_routing=True, tid2eid=tid2eid)
  got_w, got_ids = jax.jit(
      lambda pb: moe.RoutedMoE.get_topk(
          fb_host, pb, pb, input_ids=input_ids, forced_routed_experts=forced, token_mask=tmask
      )
  )(pre_bias)

  slot_valid = (np.asarray(forced) >= 0) & (np.asarray(forced) < e)
  sorted_f = np.sort(np.asarray(forced), axis=-1)
  is_forced = np.all(slot_valid, axis=-1) & ~np.any(sorted_f[..., 1:] == sorted_f[..., :-1], axis=-1)
  is_pad = ~np.asarray(tmask)
  is_fallback = ~is_forced & ~is_pad

  assert _bits_equal(np.asarray(got_ids)[is_fallback], np.asarray(unforced_ids)[is_fallback])
  assert _bits_equal(np.asarray(got_w)[is_fallback], np.asarray(unforced_w)[is_fallback])
  assert _bits_equal(np.asarray(got_ids)[is_forced & ~is_pad], np.asarray(forced)[is_forced & ~is_pad])

  with pytest.raises(ValueError, match="input_ids cannot be None when is_hash_routing is True"):
    moe.RoutedMoE.get_topk(fb_host, pre_bias, pre_bias, input_ids=None, forced_routed_experts=forced)


def _walk_jaxpr_eqns(jaxpr):
  """Yields all equations recursively across nested cond/pjit jaxprs."""
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          yield from _walk_jaxpr_eqns(sub)


def test_c4096_linear_reverse_map_and_cap_ge_num_tokens_short_circuit():
  """Verifies O(T*k) reverse gather without 3D [k, C, T] temporaries at C=4096 and short-circuit when cap >= T."""
  t, e, k, cap = 16384, 256, 8, 4096
  scores = jax.random.normal(jax.random.PRNGKey(61), (1, t, e), jnp.float32)
  uses = jnp.zeros((1, t), dtype=jnp.bool_).at[0, :12].set(True)
  topk_fn = functools.partial(moe.topk_indices_by_argmax, k=k)

  closed_jaxpr = jax.make_jaxpr(lambda s, u: moe.fallback_topk_indices_compacted(s, u, k, cap, topk_fn))(scores, uses)
  # Ensure no equation allocates a 3D [k, C, T] tensor (which would have 8 * 4096 * 16384 = 536,870,912 elements).
  max_elems = max(
      v.aval.size
      for eqn in _walk_jaxpr_eqns(closed_jaxpr.jaxpr)
      for v in list(eqn.invars) + list(eqn.outvars)
      if hasattr(v, "aval") and hasattr(v.aval, "size")
  )
  assert max_elems <= max(t * e, cap * t)

  got = np.asarray(jax.jit(lambda s, u: moe.fallback_topk_indices_compacted(s, u, k, cap, topk_fn))(scores, uses))
  want = np.asarray(jax.jit(topk_fn)(scores[0, :12]))
  assert _bits_equal(got[0, :12], want.astype(np.int32))
  assert not got[0, 12:].any()

  # When capacity >= num_tokens, fallback_topk_indices_compacted short-circuits directly to topk_fn (no cond/iota).
  short_scores = scores[:, :256, :]
  short_uses = uses[:, :256]
  for big_cap in (256, 512):
    jx_str = str(
        jax.make_jaxpr(lambda s, u, c=big_cap: moe.fallback_topk_indices_compacted(s, u, k, c, topk_fn))(
            short_scores, short_uses
        )
    )
    assert "cond[" not in jx_str
    assert "broadcasted_iota" not in jx_str

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
"""ragged_unsort_spread_masked_indices against the key off, on CPU.

With moe_routing_index_math="count" and a packed buffer, the backward of the ring-of-experts unsort sends the
routing-weight gradient back to slot order with a gather through the partial inverse permutation
(`ragged_sort._ring_unsort_fan_out`). Its masked lanes (other shards' slots and dropped slots) read table entry 0; the
key gives each masked lane its own slot index, an in-bounds entry of the padded n-entry table, instead. The outer where
discards those lanes, so everything must be bit-identical to the key off; only the addresses the gather reads change.

What a CPU run can and cannot reach: off a TPU target the ragged gather and gather-reduce take their jnp reference
fallbacks, so the SparseCore gather whose address pattern the key changes (and its speed) is TPU-only. The index math,
the gather through the padded table and the outer where are plain JAX and run here as on TPU.

In-process:
  - The key: a bool, default False in types.py and base.yml, resolved by pyconfig (and True when passed).
  - The fan-out directly, every shard of a 4-shard ring, 3 seeds x {packed buffer, packed buffer that drops, full
    buffer} x {"count", "sort"}: the gathered rows, the weights, the mask and the slot-order value bit-identical with the
    key on (it is a no-op for "sort" and for the full buffer; they are compared anyway). Again for "count" with NaN, +Inf
    and 3e38 in the per-row values outside the mask: the spread lanes now read those entries, and the result must still
    be identical.
  - Structure, two-sided: the traced fan-out has exactly one iota more with the key on (the spread index) for "count" on
    a packed buffer, and the same program for "sort" and for the full buffer. So a key that reaches the fan-out but
    leaves the index at 0 cannot pass.
  - Deliberately wrong spreads must FAIL the fan-out comparison: without the outer where, valid lanes one row off, and
    one valid slot dropped.
In child processes with 8 CPU devices (they never skip for want of devices):
  - The op: ring_ragged_sort, then ring_ragged_unsort_down_projection (the fused down projection) or ring_ragged_unsort,
    on a 4-device expert-parallel mesh, bf16, forward and VJP: the output, dx, dw, dwi and dwo bit-identical with the key
    on, for {full, packed, dropping} x {"count", "sort"} x {fused, plain} x 2 seeds; then "count" at expert parallelism 8
    (16 experts, about 7 of 8 slots masked per shard, as in production). A slot-order gradient one ulp off, and a spread
    without the outer where, must FAIL.
  - The layer: the shrunk Qwen3.5 routed MoE of moe_roe_local_routing_test.py (cp-as-ep, ring-of-experts ragged sort,
    local routing, two token chunks) with moe_routing_index_math="count": the plain unsort, the fused down projection
    (moe_router_grad_from_wo_input) and the benchmark's MoE settings. The output, the loss and every gradient
    bit-identical with the key on; every call of the two ring unsorts and of the fan-out carries the run's key value (a
    spy), each case reaches its own call site, and at least one fan-out call is "count" on a packed buffer. So a call
    site that drops the key cannot pass.
"""

import contextlib
import os
import subprocess
import sys
from unittest import mock

import jax
from jax._src import core as jax_core
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import yaml

from maxtext.configs import types as mt_types
from maxtext.kernels.ragged import ragged_sort

_KEY = "ragged_unsort_spread_masked_indices"
# The key this file tests. On a MaxText without it this line raises, so the file cannot be collected there.
_KEY_FIELD = mt_types.MoEGeneral.model_fields[_KEY]
_SPREAD = {"spread_masked_indices": True}
_NUM_DEVICES = 8
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES}"
_OP_PASS_TOKEN = "RAGGED_UNSORT_SPREAD_OP_CHECKS_PASSED"
_LAYER_PASS_TOKEN = "RAGGED_UNSORT_SPREAD_LAYER_CHECKS_PASSED"


def _bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  if a.dtype.kind == "f" or a.dtype == jnp.bfloat16:
    return bool(np.array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)))
  return bool(np.array_equal(a, b))


def test_key_is_a_bool_default_off():
  assert _KEY_FIELD.annotation is bool
  assert _KEY_FIELD.default is False
  base_yml = os.path.join(os.path.dirname(mt_types.__file__), "base.yml")
  with open(base_yml, encoding="utf-8") as f:
    assert yaml.safe_load(f)[_KEY] is False
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  common = {
      "run_name": "spread_key",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  assert getattr(pyconfig.initialize([sys.argv[0], get_test_config_path()], **common), _KEY) is False
  assert getattr(pyconfig.initialize([sys.argv[0], get_test_config_path()], **common, **{_KEY: True}), _KEY) is True


# ------------------------------------------------------------------------------------------------ the fan-out, directly
_T, _K, _E, _EP = 512, 4, 16, 4  # 2048 slots, 4 experts per shard
# (skew, buffer rows): packed (no drop), packed with the first shard's experts overloaded so its buffer drops, full.
_BUFFERS = {"packed": (False, 640), "dropping": (True, 384), "full": (False, _T * _K)}
_GATHER_KWARGS = {
    "enforce_fallback": True,
    "flops_override": -1,
    "bytes_accessed_override": -1,
    "use_single_sparsecore": False,
}


def _routing(seed, skew):
  k = jax.random.split(jax.random.PRNGKey(seed), 3)
  logits = jax.random.normal(k[0], (_T, _E))
  if skew:
    logits = logits + jnp.where(jnp.arange(_E) < _E // _EP, 2.5, 0.0)
  ids = jax.lax.top_k(logits, _K)[1].astype(jnp.int32)
  w = jax.random.uniform(k[1], (_T * _K,), jnp.float32, 0.25, 0.3)
  return ids, w, k[2]


def _maps(ids, shard, index_math):
  """The shard's window and the index maps ring_ragged_sort builds for `index_math`."""
  flat = ids.reshape(-1)
  local = _E // _EP
  offs = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(jax.nn.one_hot(flat, _E, dtype=jnp.int32).sum(0))])
  start, end = offs[shard * local], offs[(shard + 1) * local]
  if index_math == "count":
    _, idx_inv = ragged_sort.packed_stable_sort(flat, _E)
    pos = ragged_sort._local_slot_positions(ids, _E, shard, _EP)  # pylint: disable=protected-access
    revert = jnp.where(pos >= 0, start + pos, -1).astype(jnp.int32)
  else:
    idx_inv = jnp.argsort(flat, stable=True).astype(jnp.int32)
    revert = jnp.argsort(idx_inv).astype(jnp.int32)
  return start, end, idx_inv, revert


def _fan_out_fn(buffer_size, index_math, garbage=False, **kw):
  """The fan-out and its slot-order map applied to a per-row value that is non-zero inside the mask."""

  def f(g, revert, w, start, end, idx_inv):
    gathered, weights, mask, to_slot = ragged_sort._ring_unsort_fan_out(  # pylint: disable=protected-access
        g, revert, w, start, end, buffer_size, _K, idx_inv, _GATHER_KWARGS, mask_rows=False, index_math=index_math, **kw
    )
    rows = gathered.shape[0]
    good = jnp.arange(rows, dtype=jnp.float32) * 0.37 + 1.0
    if garbage:
      bad = jnp.where(jnp.arange(rows) % 3 == 0, jnp.nan, jnp.where(jnp.arange(rows) % 3 == 1, jnp.inf, 3.0e38))
      per_row = jnp.where(mask, good, bad)
    else:
      per_row = jnp.where(mask, good, 0.0)
    return gathered, weights, mask, to_slot(per_row)

  return f


def _fan_out_args(ids, w, key, shard, index_math):
  start, end, idx_inv, revert = _maps(ids, shard, index_math)
  g = jax.random.normal(key, (_T, 64), jnp.float32).astype(jnp.bfloat16)
  return g, revert, w, start, end, idx_inv


def _fan_out_cases(index_math, buffer, garbage=False, **kw):
  """(case, key off, key on) for 3 seeds x every shard; `kw` goes to the key-on run only."""
  skew, buffer_size = _BUFFERS[buffer]
  for seed in range(3):
    ids, w, key = _routing(seed, skew)
    for shard in range(_EP):
      args = _fan_out_args(ids, w, key, shard, index_math)
      off = jax.jit(_fan_out_fn(buffer_size, index_math, garbage))(*args)
      on = jax.jit(_fan_out_fn(buffer_size, index_math, garbage, **_SPREAD, **kw))(*args)
      yield f"seed{seed} shard{shard}", off, on


_CASES = [(m, b) for m in ("count", "sort") for b in ("packed", "dropping", "full")]


@pytest.mark.parametrize("index_math,buffer", _CASES, ids=[f"{m}-{b}" for m, b in _CASES])
def test_fan_out_bit_identical(index_math, buffer):
  for case, off, on in _fan_out_cases(index_math, buffer):
    names = ("gathered", "weights", "mask", "slot order")
    bad = [n for n, x, y in zip(names, off, on) if not _bits_equal(x, y)]
    assert not bad, f"{case}: {bad} differ with the key on"


@pytest.mark.parametrize("buffer", ["packed", "dropping"])
def test_fan_out_bit_identical_with_garbage_outside_the_mask(buffer):
  for case, off, on in _fan_out_cases("count", buffer, garbage=True):
    assert _bits_equal(off[3], on[3]), f"{case}: the slot-order value differs with the key on"
    assert np.isfinite(np.asarray(on[3])).all(), f"{case}: garbage leaked into the slot-order value"


def _count_eqns(jaxpr, name):
  total = 0
  for eqn in jaxpr.eqns:
    total += eqn.primitive.name == name
    for sub in jax_core.jaxprs_in_params(eqn.params):
      total += _count_eqns(sub, name)
  return total


@pytest.mark.parametrize("index_math,buffer", _CASES, ids=[f"{m}-{b}" for m, b in _CASES])
def test_spread_changes_only_the_count_packed_gather_index(index_math, buffer):
  skew, buffer_size = _BUFFERS[buffer]
  ids, w, key = _routing(0, skew)
  args = _fan_out_args(ids, w, key, 0, index_math)
  off = jax.make_jaxpr(_fan_out_fn(buffer_size, index_math))(*args)
  on = jax.make_jaxpr(_fan_out_fn(buffer_size, index_math, **_SPREAD))(*args)
  if index_math == "count" and buffer != "full":
    # The masked lanes' index: iota(n) instead of the constant 0.
    assert _count_eqns(on.jaxpr, "iota") == _count_eqns(off.jaxpr, "iota") + 1
  else:
    assert str(on) == str(off)


def _wrong_fan_out(mode):
  """The fan-out with a deliberately wrong spread in its count/packed slot-order map (key on only)."""
  real = ragged_sort._ring_unsort_fan_out  # pylint: disable=protected-access

  def fan_out(*args, **kwargs):
    spread = kwargs.pop("spread_masked_indices", False)
    out = real(*args, **kwargs)
    revert, start, end, buffer_size = args[1], args[3], args[4], args[5]
    n = revert.shape[0]
    if not spread or kwargs.get("index_math") != "count" or buffer_size >= n:
      return out
    gathered, weights, mask, _ = out
    limit = jnp.minimum(end - start, buffer_size)

    def wrong(per_row):
      shifted = revert - start
      valid = (shifted >= 0) & (shifted < limit)
      table = jnp.pad(per_row, (0, n - buffer_size))
      spread_index = jnp.arange(n, dtype=shifted.dtype)
      if mode == "no_outer_where":
        return table[jnp.where(valid, shifted, spread_index)]
      if mode == "off_by_one":
        return jnp.where(valid, table[jnp.where(valid, jnp.minimum(shifted + 1, n - 1), spread_index)], 0.0)
      valid = valid.at[jnp.argmax(valid)].set(False)  # drop_one_valid
      return jnp.where(valid, table[jnp.where(valid, shifted, spread_index)], 0.0)

    return gathered, weights, mask, wrong

  return fan_out


@pytest.mark.parametrize("mode", ["no_outer_where", "off_by_one", "drop_one_valid"])
def test_wrong_spreads_are_flagged(mode):
  flagged = []
  with mock.patch.object(ragged_sort, "_ring_unsort_fan_out", _wrong_fan_out(mode)):
    for buffer in ("packed", "dropping"):
      for _, off, on in _fan_out_cases("count", buffer):
        flagged.append(not _bits_equal(off[3], on[3]))
  assert len(flagged) == 24 and all(flagged), f"{flagged.count(True)} of {len(flagged)} flagged"


# ------------------------------------------------------------------------------------------------ child processes
def _run_child(mode, token):
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {_XLA_FLAGS}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__, mode], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert token in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.cpu_only
def test_op_forward_and_vjp_on_cpu_mesh():
  """Runs `_op_checks` in a child process with 8 CPU devices."""
  _run_child("op", _OP_PASS_TOKEN)


@pytest.mark.cpu_only
def test_routed_moe_layer_on_cpu_mesh():
  """Runs `_layer_checks` in a child process with 8 CPU devices."""
  _run_child("layer", _LAYER_PASS_TOKEN)


# ------------------------------------------------------------------------------------------------ the op (child)
_D, _F = 128, 64
_OP_NAMES = ("out", "dx", "dw", "dwi", "dwo")


def _grouped_matmul(h, w, group_sizes, expert_assignments, group_offset):
  del expert_assignments
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(h.shape[0])[:, None] >= ends[None, :], axis=1) - group_offset
  out = jnp.zeros((h.shape[0], w.shape[-1]), jnp.float32)
  for e in range(w.shape[0]):
    out = out + jnp.where((local == e)[:, None], h.astype(jnp.float32) @ w[e].astype(jnp.float32), 0.0)
  return out.astype(h.dtype)


def _expert_operands(gs, buffer_rows, num_slots, local):
  shard = jax.lax.axis_index("ep")
  if buffer_rows < num_slots:
    ends = jnp.minimum(jnp.cumsum(jax.lax.dynamic_slice_in_dim(gs, shard * local, local)), buffer_rows)
    return (ends - jnp.concatenate([jnp.zeros((1,), ends.dtype), ends[:-1]]), None, 0)
  return (gs, None, shard * local)


def _op_run(buffer_size, index_math, fused, seed=0, ep=4, experts=8, topk=2, tokens=64, **kw):
  """ring_ragged_sort -> (wi, silu) -> the unsort, forward and VJP, on an `ep`-device expert-parallel mesh."""
  k = jax.random.split(jax.random.PRNGKey(seed), 6)
  x = jax.random.normal(k[0], (tokens, _D), jnp.float32)
  ids = jnp.stack([jax.random.permutation(kk, experts)[:topk] for kk in jax.random.split(k[1], tokens)]).astype(jnp.int32)
  w = jax.nn.softmax(jax.random.normal(k[2], (tokens, topk)), axis=-1).reshape(-1)
  wi = (jax.random.normal(k[3], (experts, _D, _F)) * 0.1).reshape(ep, experts // ep, _D, _F).astype(jnp.bfloat16)
  wo = (jax.random.normal(k[4], (experts, _F, _D)) * 0.1).reshape(ep, experts // ep, _F, _D).astype(jnp.bfloat16)
  cot = jax.random.normal(k[5], (tokens, _D), jnp.float32)
  local = experts // ep
  mesh = Mesh(np.array(jax.devices()[:ep]), ("ep",))

  def body(x, w, wi, wo, cot):
    wi, wo = wi[0], wo[0]

    def f(x, w, wi, wo):
      sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
          x, ids, experts, topk, "ep", ep, buffer_size=buffer_size, index_math=index_math
      )
      operands = _expert_operands(gs, sorted_x.shape[0], tokens * topk, local)
      h = jax.nn.silu(_grouped_matmul(sorted_x, wi, *operands).astype(jnp.float32)).astype(sorted_x.dtype)
      perm_arg = perm if index_math == "count" else None
      if fused:
        return ragged_sort.ring_ragged_unsort_down_projection(
            h,
            wo,
            _grouped_matmul,
            operands,
            gs,
            revert,
            topk,
            local,
            "ep",
            topk_weights=w,
            topk_argsort_indices=perm_arg,
            index_math=index_math,
            **kw,
        )
      y = _grouped_matmul(h, wo, *operands)
      return ragged_sort.ring_ragged_unsort(
          y, gs, revert, topk, local, "ep", topk_weights=w, topk_argsort_indices=perm_arg, index_math=index_math, **kw
      )

    out, vjp = jax.vjp(f, x, w, wi, wo)
    gx, gw, gwi, gwo = vjp(cot.astype(out.dtype))
    return out[None], gx[None], gw[None], gwi[None], gwo[None]

  sm = jax.shard_map(
      body, mesh=mesh, in_specs=(P(), P(), P("ep"), P("ep"), P()), out_specs=(P("ep"),) * 5, check_vma=False
  )
  return [np.asarray(a) for a in jax.jit(sm)(x.astype(jnp.bfloat16), w, wi, wo, cot)]


def _op_case(label, buffer_size, index_math, fused, seed, control=None, **cfg):
  off = _op_run(buffer_size, index_math, fused, seed, **cfg)
  with control if control is not None else contextlib.nullcontext():
    on = _op_run(buffer_size, index_math, fused, seed, **cfg, **_SPREAD)
  bad = [n for n, a, b in zip(_OP_NAMES, off, on) if not _bits_equal(a, b)]
  print(f"{label}: {'identical' if not bad else f'DIFFERENT in {bad}'}", flush=True)
  return not bad


def _one_ulp_fan_out():
  """Deliberately wrong: with the key on, the largest slot-order value one ulp up."""
  real = ragged_sort._ring_unsort_fan_out  # pylint: disable=protected-access

  def fan_out(*args, **kwargs):
    gathered, weights, mask, to_slot = real(*args, **kwargs)
    if not kwargs.get("spread_masked_indices"):
      return gathered, weights, mask, to_slot

    def wrong(per_row):
      v = to_slot(per_row)
      i = jnp.argmax(jnp.abs(v))
      return v.at[i].set(jnp.nextafter(v[i], jnp.float32(np.inf)))

    return gathered, weights, mask, wrong

  return fan_out


def _op_checks():
  """Every op case identical with the key on; both deliberately wrong spreads detected."""
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  results = {}
  for bname, buf in (("full", 128), ("packed", 48), ("dropping", 24)):
    for index_math in ("count", "sort"):
      for fused in (True, False):
        for seed in (0, 1):
          label = f"ep4 {bname} {index_math} {'fused' if fused else 'plain'} seed{seed}"
          results[label] = _op_case(label, buf, index_math, fused, seed)
  ep8 = {"ep": 8, "experts": 16, "topk": 2, "tokens": 128}
  for bname, buf in (("packed", 48), ("dropping", 24)):
    for fused in (True, False):
      for seed in (0, 1):
        label = f"ep8 {bname} count {'fused' if fused else 'plain'} seed{seed}"
        results[label] = _op_case(label, buf, "count", fused, seed, **ep8)
  controls = {
      "control one ulp": _op_case(
          "control one ulp",
          24,
          "count",
          True,
          0,
          control=mock.patch.object(ragged_sort, "_ring_unsort_fan_out", _one_ulp_fan_out()),
      ),
      "control no outer where": _op_case(
          "control no outer where",
          24,
          "count",
          True,
          0,
          control=mock.patch.object(ragged_sort, "_ring_unsort_fan_out", _wrong_fan_out("no_outer_where")),
      ),
  }
  failed = [k for k, v in results.items() if not v]
  undetected = [k for k, v in controls.items() if v]
  print(f"op cases {len(results)}, failed {failed}; controls {len(controls)}, NOT detected {undetected}", flush=True)
  if failed or undetected:
    sys.exit(1)
  print(_OP_PASS_TOKEN, flush=True)


# ------------------------------------------------------------------------------------------------ the layer (child)
_NUM_EXPERTS, _TOP_K, _EMB, _SEQ = 16, 4, 256, 64
# The benchmark's MoE settings and every MoE key that composes with them on the megablox path the CPU runs (as in
# moe_router_grad_from_wo_input_test.py), with "count".
_BENCH = {
    "moe_chunk_pipeline": True,
    "moe_expert_weight_prefetch": "forward",
    "ring_of_experts_row_major_reduce_scatter": True,
    "moe_routing_maps": "device",
    "ragged_buffer_factor": 2.75,
    "prefuse_moe_weights": True,
    "fp8_moe_fake_quant": True,
    "moe_exclude_padding_from_routing": True,
    "moe_unsplit_prefused_wi": True,
    "moe_prefused_swiglu_vjp": True,
    "moe_roe_packed_routing_gather": True,
    "moe_router_grad_from_wo_input": True,
}
_LAYER_CASES = {
    # case: (overrides, input mode, the ring unsort it must reach)
    "plain": ({}, "real", "ring_ragged_unsort"),
    "fused": ({"moe_router_grad_from_wo_input": True}, "real", "ring_ragged_unsort_down_projection"),
    "bench": (_BENCH, "padmask", "ring_ragged_unsort_down_projection"),
}


def _layer_cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh, with "count" (moe_roe_local_routing_test.py's)."""
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  kwargs = {
      "run_name": "ragged_unsort_spread_masked_indices",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": True,
      "num_experts": _NUM_EXPERTS,
      "num_experts_per_tok": _TOP_K,
      "base_emb_dim": _EMB,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": 0.25,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "float32_gate_logits": True,
      "megablox": True,
      "sparse_matmul": True,
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": 1,
      "ici_context_parallelism": 4,
      "ici_expert_parallelism": 2,
      "context_parallel_load_balance": False,
      "ring_of_experts_local_routing": True,
      "moe_routing_index_math": "count",
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _layer_inputs(cfg, mode):
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  token_mask = None
  if mode == "padmask":
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, token_mask


def _layer_run(cfg, mesh, params, x, token_mask):
  """Loss, output and gradients, and the spread value of every ring unsort and fan-out call traced."""
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  from maxtext.layers import moe  # pylint: disable=import-outside-toplevel
  from maxtext.layers.initializers import nd_dense_init  # pylint: disable=import-outside-toplevel

  calls = []

  def spy(name, real):
    def wrapped(*args, **kwargs):
      if name == "fan_out":
        packed = args[5] < args[1].shape[0]
        calls.append(("fan_out", kwargs.get("spread_masked_indices", False), kwargs.get("index_math"), packed))
      else:
        calls.append((name, kwargs.get("spread_masked_indices", False), kwargs.get("index_math"), None))
      return real(*args, **kwargs)

    return wrapped

  patches = [
      mock.patch.object(moe, "ring_ragged_unsort", spy("ring_ragged_unsort", moe.ring_ragged_unsort)),
      mock.patch.object(
          moe,
          "ring_ragged_unsort_down_projection",
          spy("ring_ragged_unsort_down_projection", moe.ring_ragged_unsort_down_projection),
      ),
      mock.patch.object(
          ragged_sort,
          "_ring_unsort_fan_out",
          spy("fan_out", ragged_sort._ring_unsort_fan_out),  # pylint: disable=protected-access
      ),
  ]
  with contextlib.ExitStack() as stack:
    for p in patches:
      stack.enter_context(p)
    stack.enter_context(jax.set_mesh(mesh))
    stack.enter_context(nn_partitioning.axis_rules(cfg.logical_axis_rules))
    layer = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )
    graphdef, _, rest = nnx.split(layer, nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), calls


def _layer_params(cfg, mesh):
  """The layer's parameters, built once and shared by the key-off and key-on runs."""
  from flax import nnx  # pylint: disable=import-outside-toplevel
  from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
  from maxtext.layers import moe  # pylint: disable=import-outside-toplevel
  from maxtext.layers.initializers import nd_dense_init  # pylint: disable=import-outside-toplevel

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    layer = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.moe_mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )
    _, params, _ = nnx.split(layer, nnx.Param, ...)
  return params


def _layer_case(label, overrides, mode, call_site):
  """The layer with the key off and on, on the same parameters: bit-identical, and the key reaches `call_site`."""
  from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

  cfg_off = _layer_cfg(**overrides, **{_KEY: False})
  cfg_on = _layer_cfg(**overrides, **{_KEY: True})
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_off), cfg_off.mesh_axes)
  x, token_mask = _layer_inputs(cfg_off, mode)
  params = _layer_params(cfg_off, mesh)
  (loss_off, out_off, grads_off), calls_off = _layer_run(cfg_off, mesh, params, x, token_mask)
  (loss_on, out_on, grads_on), calls_on = _layer_run(cfg_on, mesh, params, x, token_mask)
  passed, lines = True, []
  for name, a, b in [("output", out_off, out_on), ("loss", loss_off, loss_on)] + [
      (jax.tree_util.keystr(path), a, b)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_off)[0], jax.tree_util.tree_leaves(grads_on))
  ]:
    same = _bits_equal(a, b) and bool(np.isfinite(np.asarray(b, np.float32)).all())
    passed &= same
    lines.append(f"  [{'ok' if same else 'FAIL'}] {name}: bit-identical={same}")
  for value, calls in ((False, calls_off), (True, calls_on)):
    sites = sorted({c[0] for c in calls if c[0] != "fan_out"})
    carried = bool(calls) and all(c[1] is value for c in calls)
    reached = sites == [call_site]
    count_packed = any(c[0] == "fan_out" and c[2] == "count" and c[3] for c in calls)
    ok = carried and reached and count_packed
    passed &= ok
    lines.append(
        f"  [{'ok' if ok else 'FAIL'}] key={value}: {len(calls)} calls carry it={carried}, call sites {sites}"
        f" (want [{call_site}]), a count fan-out on a packed buffer={count_packed}"
    )
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _layer_checks():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  results = {name: _layer_case(name, *case) for name, case in _LAYER_CASES.items()}
  failed = [k for k, v in results.items() if not v]
  print(f"layer cases {len(results)}, failed {failed}", flush=True)
  if failed:
    sys.exit(1)
  print(_LAYER_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  {"op": _op_checks, "layer": _layer_checks}[sys.argv[1]]()

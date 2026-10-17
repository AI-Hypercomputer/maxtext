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
"""moe_fp8_token_all_gather on a CPU mesh: the ring-of-experts token all-gather with an fp8 payload.

With the key, each expert shard quantizes its tokens per row (e4m3, one f32 absmax scale per row), all-gathers the fp8
codes and the scales, and dequantizes the gathered rows to bfloat16 right after the gather. The backward is the plain
reduce-scatter of the cotangent (straight-through).

Expected error, stated before measuring: e4m3 keeps 3 mantissa bits, so one rounding has a relative error of at most
2^-4 = 6.25% and an RMS of ~2.65% for mantissas spread like real data; with an absmax row scale every value of a row stays
in the normal range unless it is ~2^15 times smaller than the row's max, so the per-row relative L2 error of the
dequantized tokens is ~2.65% (median band 1.5%..3.5%, max <= 6%). The backward of the gather alone is bit-identical to the
plain all-gather's (the same reduce-scatter of the same cotangent).

Runs as a subprocess so the forced device count takes effect before JAX initializes. The mesh is 8 CPU devices,
fsdp1 x context4 x expert2 under cp-as-ep, so the expert-parallel group is ("context", "expert") of size 8, as in the
Qwen3.5-397B configuration (fsdp32 x context4 x expert2).

Checks (each with deliberately broken positive controls that must fail):
  1. The helper `ep_fp8_token_all_gather` against the plain all-gather: forward per-row error in the e4m3 band, zero rows
     exact, backward bit-identical. Broken: dequantization without the scale, every shard's rows scaled by the local
     scale (scales not gathered), a backward without the reduce.
  2. The routed-MoE layer on the bench stack (pipelined chunk loop, two token chunks, moe_router_grad_from_wo_input,
     moe_wi_gmm_fused_swiglu, moe_wo_gmm_fused_dswiglu, packed routing gather, count index math, padding excluded), with
     the jnp stand-ins of moe_wo_gmm_fused_dswiglu_test.py for the two Pallas GMM entry points and the kernel module's
     own fused epilogue: key off twice bit-identical; key on changes the output at the e4m3 level, keeps the loss and
     every gradient close, and the traced program has e4m3 all-gathers only with the key. Broken: a gather that drops
     the scale must fail.
"""

import functools
import os
import re
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_PASS_TOKEN = "MOE_FP8_TOKEN_ALL_GATHER_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_EP_AXES = ("context", "expert")
# Pre-registered bands (see the module docstring).
_E4M3_MEDIAN_BAND = (0.015, 0.035)
_E4M3_ROW_MAX = 0.06
_LAYER_MEDIAN_BAND = (0.010, 0.040)
_LAYER_MAX = 0.10
_GRAD_REL_MAX = 0.10
_BROKEN_MIN = 0.5
_REAL_HELPER = moe.ep_fp8_token_all_gather


@pytest.mark.cpu_only
def test_moe_fp8_token_all_gather_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {_XLA_FLAGS}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _row_rel(ref, new):
  """Per-row relative L2 error (rows = all leading axes, the last axis is the row)."""
  ref = np.asarray(ref, np.float64).reshape(-1, np.shape(ref)[-1])
  new = np.asarray(new, np.float64).reshape(-1, np.shape(new)[-1])
  norm = np.linalg.norm(ref, axis=-1)
  keep = norm > 0
  return np.linalg.norm(new - ref, axis=-1)[keep] / norm[keep]


def _stats(r):
  return float(np.median(r)), float(np.quantile(r, 0.99)), float(np.max(r))


def _in(v, band):
  return band[0] <= v <= band[1]


# ---------------------------------------------------------------------------------------------------------------
# Deliberately broken helpers (positive controls).
def _broken_no_scale(x, axis_name, all_gather_fn, qtype=jnp.float8_e4m3fn):
  """Broken on purpose: the receiver dequantizes without the row scale."""
  del axis_name
  q, _ = moe.quantize_rows_fp8(x, qtype)
  return all_gather_fn(q).astype(x.dtype)


def _broken_local_scale(x, axis_name, all_gather_fn, qtype=jnp.float8_e4m3fn):
  """Broken on purpose: the scales are not gathered; every gathered row is scaled by this shard's row scale."""
  n = jax.lax.axis_size(axis_name)
  q, s = moe.quantize_rows_fp8(x, qtype)
  return moe.dequantize_rows_fp8(all_gather_fn(q), jnp.concatenate([s] * n, axis=0), x.dtype)


def _broken_bwd_no_reduce(x, axis_name, all_gather_fn, qtype=jnp.float8_e4m3fn):
  """Broken on purpose: the backward takes this shard's slice of the cotangent without summing over the shards."""

  @jax.custom_vjp
  def f(z):
    q, s = moe.quantize_rows_fp8(z, qtype)
    return moe.dequantize_rows_fp8(all_gather_fn(q), all_gather_fn(s), z.dtype)

  def f_fwd(z):
    return f(z), None

  def f_bwd(_, g):
    i = jax.lax.axis_index(axis_name)
    rows = g.shape[0] // jax.lax.axis_size(axis_name)
    return (jax.lax.dynamic_slice_in_dim(g, i * rows, rows, axis=0),)

  f.defvjp(f_fwd, f_bwd)
  return f(x)


# ---------------------------------------------------------------------------------------------------------------
# 1. The helper on a ("context", "expert") = (4, 2) mesh.
def _helper_checks():
  mesh = jax.make_mesh((4, 2), _EP_AXES)
  n, b, s, e = _NUM_DEVICES, 1, 64, 256
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
  # Tokens [n * b, s, e] (shard i owns batch row i) with row norms spread over ~3 decades, and some all-zero rows.
  x = jax.random.normal(k1, (n * b, s, e), jnp.float32) * jnp.exp(2.0 * jax.random.normal(k2, (n * b, s, 1)))
  x = x.at[:, 3, :].set(0.0).astype(jnp.bfloat16)
  w = jax.random.normal(k3, (n, n * b, s, e), jnp.float32)  # cotangent probe: each shard's gathered copy

  def gather_plain(z):
    return jax.lax.all_gather(z, axis_name=_EP_AXES, tiled=True)

  xs = jax.device_put(x, NamedSharding(mesh, P(_EP_AXES)))

  def fwd(helper):
    f = jax.shard_map(
        lambda z: helper(z, _EP_AXES, gather_plain)[None],
        mesh=mesh,
        in_specs=(P(_EP_AXES),),
        out_specs=P(_EP_AXES),
        check_vma=False,
    )
    return np.asarray(jax.jit(f)(xs).astype(jnp.float32))  # [n, n * b, s, e]: every shard's gathered copy

  results = {}
  plain = fwd(lambda z, a, g: g(z))
  exact = np.broadcast_to(np.asarray(x, np.float32)[None], plain.shape)
  results["H0_plain_gather_exact"] = bool(np.array_equal(plain, exact))
  got = fwd(_REAL_HELPER)
  r = _row_rel(exact, got)
  med, p99, mx = _stats(r)
  zero_rows_exact = bool(np.array_equal(got[:, :, 3, :], np.zeros_like(got[:, :, 3, :])))
  ok = _in(med, _E4M3_MEDIAN_BAND) and mx <= _E4M3_ROW_MAX and zero_rows_exact and bool(np.isfinite(got).all())
  print(
      f"H1 fp8 gather vs exact tokens: per-row median {med:.4%} p99 {p99:.4%} max {mx:.4%} (band median"
      f" {_E4M3_MEDIAN_BAND}, max <= {_E4M3_ROW_MAX}); zero rows exact {zero_rows_exact} -> {'ok' if ok else 'FAIL'}",
      flush=True,
  )
  results["H1_fp8_fwd"] = ok
  for name, helper in (("H2_broken_no_scale", _broken_no_scale), ("H3_broken_local_scale", _broken_local_scale)):
    med_b = _stats(_row_rel(exact, fwd(helper)))[0]
    detected = med_b > _BROKEN_MIN
    print(f"{name}: median {med_b:.4%} (must exceed {_BROKEN_MIN:.0%}) -> {'detected' if detected else 'NOT DETECTED'}")
    results[name] = detected

  def grad(helper):
    def loss(z):
      out = jax.shard_map(
          lambda t: helper(t, _EP_AXES, gather_plain)[None],
          mesh=mesh,
          in_specs=(P(_EP_AXES),),
          out_specs=P(_EP_AXES),
          check_vma=False,
      )(z)
      return jnp.sum(out.astype(jnp.float32) * w)

    return np.asarray(jax.jit(jax.grad(loss))(xs).astype(jnp.float32))

  g_plain = grad(lambda z, a, g: g(z))
  g_fp8 = grad(_REAL_HELPER)
  same = bool(np.array_equal(g_fp8, g_plain)) and float(np.max(np.abs(g_plain))) > 0
  print(f"H4 backward == plain all-gather's backward bit for bit: {same} -> {'ok' if same else 'FAIL'}", flush=True)
  results["H4_bwd_identical"] = same
  g_bad = grad(_broken_bwd_no_reduce)
  detected = not np.array_equal(g_bad, g_plain)
  print(f"H5_broken_bwd_no_reduce: differs from the plain backward: {detected} -> {'detected' if detected else 'NOT'}")
  results["H5_broken_bwd_no_reduce"] = detected
  return results


# ---------------------------------------------------------------------------------------------------------------
# 2. The routed-MoE layer on the bench stack, with the stand-ins of moe_wo_gmm_fused_dswiglu_test.py.
def _bench_stack():
  from tests.unit import moe_wo_gmm_fused_dswiglu_test as dsw  # pylint: disable=import-outside-toplevel

  stack = {
      "moe_chunk_pipeline": True,
      "moe_router_grad_from_wo_input": True,
      "moe_exclude_padding_from_routing": True,
      "moe_routing_index_math": "count",
      "moe_unsplit_prefused_wi": True,
      "moe_roe_packed_routing_gather": True,
      "moe_gmm_v2_dlhs_transpose_rhs": True,
      "moe_wi_gmm_fused_swiglu": True,
      "moe_wo_gmm_fused_dswiglu": True,
      "wi_tile_fwd_mlp_dim": dsw._MLP,  # two forward n tiles per half, as at the 397B cell  # pylint: disable=protected-access
      "ring_of_experts_row_major_reduce_scatter": True,
  }
  return dsw, stack


def _run_layer(dsw, cfg, mesh, params, x, token_mask):
  """(loss, out, grads) with the stand-ins and the kernel module's own epilogue, and the traced program's text."""
  # pylint: disable=protected-access
  dsw._TRACED.clear()
  with (
      mock.patch.object(
          dsw.gmm_v2, "gmm_v2", functools.partial(dsw._gmm_v2_stand_in, epilogue=dsw._kernel_epilogue, reader=None)
      ),
      mock.patch.object(dsw.tgmm_v2, "tgmm_v2", dsw._tgmm_v2_stand_in),
      jax.set_mesh(mesh),
      nn_partitioning.axis_rules(cfg.logical_axis_rules),
  ):
    graphdef, _, rest = nnx.split(dsw._build(cfg, mesh), nnx.Param, ...)

    def loss_fn(p, xx):
      out, lb_loss, _ = nnx.merge(graphdef, p, rest)(xx, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    text = str(jax.make_jaxpr(grad_fn)(params, x))
    traced = dict(dsw._TRACED)  # the fused kernel calls of ONE trace (the jit below traces again)
    (loss, out), grads = jax.jit(grad_fn)(params, x)
  return jax.device_get((loss, out, grads)), text, traced


def _leaves(grads):
  return [(jax.tree_util.keystr(p), np.asarray(v, np.float32)) for p, v in jax.tree_util.tree_flatten_with_path(grads)[0]]


def _bit_identical(a, b):
  (la, oa, ga), (lb, ob, gb) = a, b
  same = np.array_equal(np.asarray(la), np.asarray(lb)) and np.array_equal(
      np.asarray(oa, np.float32), np.asarray(ob, np.float32)
  )
  for (_, u), (_, v) in zip(_leaves(ga), _leaves(gb)):
    same &= bool(np.array_equal(u, v))
  return bool(same)


def _close_report(label, ref, new):
  """Per-row output error, loss and per-leaf gradient relative errors; returns pass."""
  (lr, orf, gr), (ln, onw, gn) = ref, new
  orf, onw = np.asarray(orf, np.float32), np.asarray(onw, np.float32)
  r = _row_rel(orf, onw)
  med, p99, mx = _stats(r) if r.size else (0.0, 0.0, 0.0)
  out_same = bool(np.array_equal(orf, onw))
  ok = (not out_same) and _in(med, _LAYER_MEDIAN_BAND) and mx <= _LAYER_MAX
  loss_rel = abs(float(ln) - float(lr)) / abs(float(lr))
  ok &= loss_rel < 0.01 and np.isfinite(float(ln))
  lines = [
      f"  out per-row rel: median {med:.4%} p99 {p99:.4%} max {mx:.4%} bit-identical={out_same}"
      f" (want changed, median in {_LAYER_MEDIAN_BAND}, max <= {_LAYER_MAX})",
      f"  loss {float(lr):.8e} -> {float(ln):.8e} rel {loss_rel:.3e} (want < 1e-2)",
  ]
  any_grad_changed = False
  for (name, a), (_, b) in zip(_leaves(gr), _leaves(gn)):
    rel = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
    any_grad_changed |= not np.array_equal(a, b)
    good = rel < _GRAD_REL_MAX and bool(np.isfinite(b).all())
    ok &= good
    lines.append(f"  [{'ok' if good else 'FAIL'}] grad {name}: rel L2 {rel:.4%}")
  ok &= any_grad_changed
  print(f"{label}: {'PASS' if ok else 'FAIL'}\n" + "\n".join(lines), flush=True)
  return ok


def _e4m3_gathers(text):
  """Number of all_gather equations with an e4m3 operand in a jaxpr text."""
  return len(re.findall(r"f8_e4m3fn\[[^\]]*\] = all_gather", text))


def _layer_checks():
  # pylint: disable=protected-access
  dsw, stack = _bench_stack()
  results = {}
  cfg_off = dsw._cfg(**stack)
  cfg_on = dsw._cfg(**stack, moe_fp8_token_all_gather=True)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_off), cfg_off.mesh_axes)
  x, token_mask = dsw._inputs(cfg_off, "padmask")
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_off.logical_axis_rules):
    _, params, _ = nnx.split(dsw._build(cfg_off, mesh), nnx.Param, ...)
  ref, text_off, traced_off = _run_layer(dsw, cfg_off, mesh, params, x, token_mask)
  ref2, _, _ = _run_layer(dsw, cfg_off, mesh, params, x, token_mask)
  results["L0_off_deterministic"] = _bit_identical(ref, ref2)
  print(f"L0 key off twice bit-identical: {results['L0_off_deterministic']}", flush=True)

  new, text_on, traced_on = _run_layer(dsw, cfg_on, mesh, params, x, token_mask)
  results["L1_fp8_token_all_gather"] = _close_report("L1 moe_fp8_token_all_gather (bench stack)", ref, new)
  ag_off, ag_on = _e4m3_gathers(text_off), _e4m3_gathers(text_on)
  chunks = cfg_on.num_moe_token_chunks
  results["L1_structure"] = ag_off == 0 and ag_on >= chunks
  print(f"L1 structure: e4m3 all_gathers off/on {ag_off}/{ag_on} (want 0 / >= {chunks}) -> {results['L1_structure']}")
  fused_ok = all(
      t.get("fused_dlhs", 0) == chunks and t.get("fused_wi_concat", 0) == chunks for t in (traced_off, traced_on)
  )
  results["L1_fused_kernels_traced"] = fused_ok
  print(
      f"L1 fused kernels traced: off {traced_off} on {traced_on} (want fused_dlhs = fused_wi_concat = {chunks} in both)"
  )

  with mock.patch.object(moe, "ep_fp8_token_all_gather", _broken_no_scale):
    bad, _, _ = _run_layer(dsw, cfg_on, mesh, params, x, token_mask)
  med_b = _stats(_row_rel(np.asarray(ref[1], np.float32), np.asarray(bad[1], np.float32)))[0]
  results["L2_broken_no_scale"] = med_b > _BROKEN_MIN
  print(f"L2 broken gather (scale dropped): out median {med_b:.4%} -> {'detected' if med_b > _BROKEN_MIN else 'NOT'}")
  with mock.patch.object(moe, "ep_fp8_token_all_gather", _broken_local_scale):
    bad, _, _ = _run_layer(dsw, cfg_on, mesh, params, x, token_mask)
  med_b = _stats(_row_rel(np.asarray(ref[1], np.float32), np.asarray(bad[1], np.float32)))[0]
  results["L3_broken_local_scale"] = med_b > _BROKEN_MIN
  print(f"L3 broken gather (local scale): out median {med_b:.4%} -> {'detected' if med_b > _BROKEN_MIN else 'NOT'}")
  return results


class ConfigTest(absltest.TestCase):
  """The key defaults off; the validator accepts it on the bench stack and rejects what it cannot run with."""

  def test_defaults_off(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="moe_fp8_token_all_gather_config",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
    )
    self.assertFalse(cfg.moe_fp8_token_all_gather)

  def test_accepted_on_the_bench_stack(self):
    # moe_router_grad_from_wo_input, moe_wi_gmm_fused_swiglu and moe_wo_gmm_fused_dswiglu all on (their validators
    # reject moe_quantize_token_all_gather, not this key).
    dsw, stack = _bench_stack()
    cfg = dsw._cfg(**stack, moe_fp8_token_all_gather=True)  # pylint: disable=protected-access
    self.assertTrue(cfg.moe_fp8_token_all_gather)
    self.assertTrue(cfg.moe_router_grad_from_wo_input and cfg.moe_wi_gmm_fused_swiglu and cfg.moe_wo_gmm_fused_dswiglu)

  def test_rejected(self):
    dsw, _ = _bench_stack()
    for overrides, requirement in (
        ({"use_ring_of_experts": False}, "use_ring_of_experts=True"),
        ({"moe_pin_sparse_core_all_gathers": True}, "moe_pin_sparse_core_all_gathers=False"),
        ({"moe_pin_sparse_core_ep_all_gathers": True}, "moe_pin_sparse_core_ep_all_gathers=False"),
        ({"num_moe_emb_chunks": 2}, "num_moe_emb_chunks=0"),
    ):
      with self.subTest(overrides=overrides):
        # The message must come from this key's validator, not from another key's.
        with self.assertRaisesRegex(ValueError, re.escape("moe_fp8_token_all_gather=True requires:") + ".*" + re.escape(requirement)):
          dsw._cfg(moe_fp8_token_all_gather=True, **overrides)  # pylint: disable=protected-access


class HelperUnitTest(absltest.TestCase):
  """Single-device sanity: a size-1 axis gathers nothing, so the helper is the per-row fp8 roundtrip."""

  def test_single_shard_roundtrip(self):
    mesh = jax.make_mesh((1,), ("expert",))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 16, 128), jnp.float32).astype(jnp.bfloat16)

    def gather(z):
      return jax.lax.all_gather(z, axis_name="expert", tiled=True)

    f = jax.jit(
        jax.shard_map(
            lambda z: moe.ep_fp8_token_all_gather(z, "expert", gather),
            mesh=mesh,
            in_specs=(P("expert"),),
            out_specs=P("expert"),
            check_vma=False,
        )
    )
    got = np.asarray(f(x).astype(jnp.float32))
    q, s = moe.quantize_rows_fp8(x, jnp.float8_e4m3fn)
    want = np.asarray(moe.dequantize_rows_fp8(q, s, jnp.bfloat16).astype(jnp.float32))
    np.testing.assert_array_equal(got, want)
    self.assertFalse(np.array_equal(got, np.asarray(x, np.float32)))


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  results = {}
  results.update(_helper_checks())
  results.update(_layer_checks())
  print("SUMMARY", results, flush=True)
  failed = [k for k, v in results.items() if not v]
  if failed:
    print(f"FAILED: {failed}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()

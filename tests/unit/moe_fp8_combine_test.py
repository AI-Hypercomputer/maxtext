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
"""moe_fp8_combine, moe_fp8_bwd_dispatch and moe_combine_per_chunk on a CPU mesh.

The three keys come from the MLPerf v6.1 DeepSeek-V3 code (6639e3166). moe_fp8_combine replaces the ring-of-experts
combine reduce-scatter by an fp8 (e4m3, one f32 absmax scale per row) all-to-all and a local f32 sum;
moe_fp8_bwd_dispatch sends the backward all-gather of the combine cotangent as fp8 with a per-row scale;
moe_combine_per_chunk orders chunk c's unpermute after chunk c-1's reduce-scatter (math unchanged).

Expected error, stated before measuring: e4m3 keeps 3 mantissa bits, so rounding a value to the nearest e4m3 has a
relative error of at most 2^-4 = 6.25% and, for mantissas spread like real data, an RMS of
2^-3 * sqrt(E[1/m^2] / 12) = 2.65% (m log-uniform in [1, 2)). With an absmax row scale every value of a row stays in
the normal range unless it is ~2^15 times smaller than the row's max, so the per-row relative L2 error of one fp8
rounding is ~2.65%; summing the rounded partials of independent shards keeps that ratio. e5m2 (2 mantissa bits) doubles
it (~5.3%).

Runs as a subprocess so the forced device count takes effect before JAX initializes. The mesh is 8 CPU devices, fsdp1 x
context4 x expert2 under cp-as-ep, so the expert-parallel group is ("context", "expert") of size 8, as in the
Qwen3.5-397B configuration (fsdp32 x context4 x expert2).

Checks (each with a deliberately broken positive control that must fail):
  1. The helper `ep_combine_reduce_scatter` against the exact f32 sum: fp8 forward per-row error in the e4m3 band;
     a dequantization that drops the scale and an all-to-all split on the wrong axis must fail. With only the
     forward in fp8, the backward must equal the plain all-gather bit for bit. The fp8 backward (e4m3 and e5m2) per-row
     error in band; a backward that drops the scale must fail.
  2. The routed-MoE layer on the bench path (two token chunks, pipelined chunk loop, moe_router_grad_from_wo_input):
     moe_fp8_combine changes the output at the e4m3 level and keeps the loss and every gradient close;
     moe_fp8_bwd_dispatch keeps the output bit-identical and the gradients close; a broken combine must fail.
  3. moe_combine_per_chunk on the plain chunk loop: output, loss and every gradient bit-identical, and the traced
     forward carries one 'moe_combine_order' tie per chunk boundary (none with the key off). On the pipelined loop the key
     is a no-op (its tie 3 already orders the chunks): the traced program has no such tie.
"""

import os
import re
import subprocess
import sys
from types import SimpleNamespace
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
from maxtext.configs import types
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "MOE_FP8_COMBINE_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_EP_AXES = ("context", "expert")
# Pre-registered bands (see the module docstring): per-row relative L2 error of one e4m3 rounding ~2.65%.
_E4M3_MEDIAN_BAND = (0.015, 0.035)
_E4M3_MAX = 0.06
_E5M2_MEDIAN_BAND = (0.03, 0.07)
_LAYER_MEDIAN_BAND = (0.015, 0.035)
_LAYER_MAX = 0.10
_GRAD_REL_MAX = 0.10
_BROKEN_MIN = 0.5
_REAL_HELPER = moe.ep_combine_reduce_scatter


@pytest.mark.cpu_only
def test_moe_fp8_combine_on_cpu_mesh():
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
def _broken_no_scale(x, axis_name, all_gather_fn=None, bwd_qtype=None, fwd_qtype=None, all_to_all_fn=None):
  """Broken on purpose: the receiver dequantizes without the row scale."""
  del all_gather_fn, bwd_qtype, all_to_all_fn
  n = jax.lax.axis_size(axis_name)
  q, _ = moe.quantize_rows_fp8(x, fwd_qtype)
  parts = jax.lax.all_to_all(q, axis_name, 0, 0, tiled=True).astype(jnp.float32)
  return jnp.sum(parts.reshape((n, x.shape[0] // n) + x.shape[1:]), axis=0).astype(x.dtype)


def _broken_wrong_axis(x, axis_name, all_gather_fn=None, bwd_qtype=None, fwd_qtype=None, all_to_all_fn=None):
  """Broken on purpose: the all-to-all splits the sequence axis instead of the shard-major batch axis."""
  del all_gather_fn, bwd_qtype, all_to_all_fn
  n = jax.lax.axis_size(axis_name)
  q, s = moe.quantize_rows_fp8(x, fwd_qtype)
  parts = jax.lax.all_to_all(q, axis_name, 1, 0, tiled=True).astype(jnp.float32) * jax.lax.all_to_all(
      s, axis_name, 1, 0, tiled=True
  )
  return jnp.sum(parts.reshape((n, x.shape[0] // n) + x.shape[1:]), axis=0).astype(x.dtype)


def _broken_bwd_no_scale(x, axis_name, all_gather_fn=None, bwd_qtype=None, fwd_qtype=None, all_to_all_fn=None):
  """Broken on purpose: the fp8 backward all-gather drops the row scale on dequantization."""
  del all_gather_fn, fwd_qtype, all_to_all_fn

  @jax.custom_vjp
  def f(z):
    return jax.lax.psum_scatter(z, axis_name, scatter_dimension=0, tiled=True)

  def f_fwd(z):
    return f(z), None

  def f_bwd(_, g):
    q, _ = moe.quantize_rows_fp8(g, bwd_qtype)
    return (jax.lax.all_gather(q, axis_name=axis_name, tiled=True).astype(g.dtype),)

  f.defvjp(f_fwd, f_bwd)
  return f(x)


# ---------------------------------------------------------------------------------------------------------------
# 1. The helper on a ("context", "expert") = (4, 2) mesh.
def _helper_checks():
  mesh = jax.make_mesh((4, 2), _EP_AXES)
  n, b, s, e = _NUM_DEVICES, 2, 32, _EMB
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
  # Per shard: [n * b, s, e] partial sums with row norms spread over ~3 decades (log-normal row scale).
  x = jax.random.normal(k1, (n, n * b, s, e), jnp.float32) * jnp.exp(2.0 * jax.random.normal(k2, (n, n * b, s, 1)))
  x = x.astype(jnp.bfloat16)
  w = jax.random.normal(k3, (n * b, s, e), jnp.float32)  # cotangent probe, [shards x b, s, e] global

  def smap(f):
    return jax.jit(
        jax.shard_map(f, mesh=mesh, in_specs=(P(_EP_AXES),), out_specs=P(_EP_AXES), check_vma=False)
    )

  # shard i owns rows [i * n * b, (i + 1) * n * b)
  flat = jax.device_put(x.reshape(n * n * b, s, e), NamedSharding(mesh, P(_EP_AXES)))
  exact = jnp.sum(x.astype(jnp.float32).reshape(n, n, b, s, e), axis=0).reshape(n * b, s, e)

  def run(helper=_REAL_HELPER, **kw):
    return np.asarray(smap(lambda z: helper(z, _EP_AXES, **kw))(flat).astype(jnp.float32))

  results = {}
  rs_bf16 = run()
  r_bf16 = _row_rel(exact, rs_bf16)
  fp8 = run(fwd_qtype=jnp.dtype(jnp.float8_e4m3fn))
  r_fp8 = _row_rel(exact, fp8)
  med, p99, mx = _stats(r_fp8)
  ok = _in(med, _E4M3_MEDIAN_BAND) and mx <= _E4M3_MAX
  print(
      f"H1 fp8 fwd vs exact f32 sum: median {med:.4%} p99 {p99:.4%} max {mx:.4%} (band median"
      f" {_E4M3_MEDIAN_BAND}, max <= {_E4M3_MAX}); bf16 reduce-scatter vs exact: median {_stats(r_bf16)[0]:.4%}"
      f" max {_stats(r_bf16)[2]:.4%} -> {'ok' if ok else 'FAIL'}",
      flush=True,
  )
  results["H1_fp8_fwd"] = ok
  for name, helper in (("H2_broken_no_scale", _broken_no_scale), ("H3_broken_wrong_axis", _broken_wrong_axis)):
    med_b = _stats(_row_rel(exact, run(helper, fwd_qtype=jnp.dtype(jnp.float8_e4m3fn))))[0]
    detected = med_b > _BROKEN_MIN
    print(f"{name}: median {med_b:.4%} (must exceed {_BROKEN_MIN:.0%}) -> {'detected' if detected else 'NOT DETECTED'}")
    results[name] = detected

  def grad(helper=_REAL_HELPER, **kw):
    def loss(z):
      out = jax.shard_map(
          lambda t: helper(t, _EP_AXES, **kw), mesh=mesh, in_specs=(P(_EP_AXES),), out_specs=P(_EP_AXES), check_vma=False
      )(z)
      return jnp.sum(out.astype(jnp.float32) * w)

    return np.asarray(jax.jit(jax.grad(loss))(flat).astype(jnp.float32))

  g_ref = grad()
  g_exact = np.tile(np.asarray(w, np.float32), (n, 1, 1))  # d/dz sum(rs(z) * w) = all_gather(w)
  same = bool(np.array_equal(grad(fwd_qtype=jnp.dtype(jnp.float8_e4m3fn)), g_ref))
  print(f"H4 fwd-only fp8: backward == plain all-gather bit for bit: {same} -> {'ok' if same else 'FAIL'}", flush=True)
  results["H4_fwd_only_bwd_identical"] = same
  for name, qt, band in (
      ("H5_bwd_e4m3", jnp.float8_e4m3fn, _E4M3_MEDIAN_BAND),
      ("H5_bwd_e5m2", jnp.float8_e5m2, _E5M2_MEDIAN_BAND),
  ):
    med_g, p99_g, mx_g = _stats(_row_rel(g_exact, grad(bwd_qtype=jnp.dtype(qt))))
    ok = _in(med_g, band)
    print(f"{name}: grad per-row median {med_g:.4%} p99 {p99_g:.4%} max {mx_g:.4%} (band {band}) -> {'ok' if ok else 'FAIL'}")
    results[name] = ok
  med_b = _stats(_row_rel(g_exact, grad(_broken_bwd_no_scale, bwd_qtype=jnp.dtype(jnp.float8_e4m3fn))))[0]
  results["H6_broken_bwd_no_scale"] = med_b > _BROKEN_MIN
  print(f"H6_broken_bwd_no_scale: median {med_b:.4%} -> {'detected' if med_b > _BROKEN_MIN else 'NOT DETECTED'}")
  return results


# ---------------------------------------------------------------------------------------------------------------
# 2./3. The routed-MoE layer.
def _cfg(**overrides):
  kwargs = {
      "run_name": "moe_fp8_combine",
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
      "ring_of_experts_row_major_reduce_scatter": True,
      "use_random_routing": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _build(cfg, mesh):
  return moe.RoutedMoE(
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


def _run(cfg, mesh, params, x):
  """(loss, out, grads) and the traced forward's jaxpr text."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(p, xx):
      model = nnx.merge(graphdef, p, rest)
      out, lb_loss, _ = model(xx)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    text = str(jax.make_jaxpr(loss_fn)(params, x))
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), text


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


def _close_report(label, ref, new, out_should_change):
  """Per-row output error, loss and per-leaf gradient relative errors; returns pass."""
  (lr, orf, gr), (ln, onw, gn) = ref, new
  orf, onw = np.asarray(orf, np.float32), np.asarray(onw, np.float32)
  r = _row_rel(orf, onw)
  med, p99, mx = _stats(r) if r.size else (0.0, 0.0, 0.0)
  out_same = bool(np.array_equal(orf, onw))
  if out_should_change:
    ok = (not out_same) and _in(med, _LAYER_MEDIAN_BAND) and mx <= _LAYER_MAX
  else:
    ok = out_same
  loss_rel = abs(float(ln) - float(lr)) / abs(float(lr))
  ok &= loss_rel < 0.01 and np.isfinite(float(ln))
  lines = [
      f"  out per-row rel: median {med:.4%} p99 {p99:.4%} max {mx:.4%} bit-identical={out_same}"
      f" (want {'changed, median in ' + str(_LAYER_MEDIAN_BAND) if out_should_change else 'bit-identical'})",
      f"  loss {float(lr):.8e} -> {float(ln):.8e} rel {loss_rel:.3e} (want < 1e-2)",
  ]
  any_grad_changed = False
  for (name, a), (_, b) in zip(_leaves(gr), _leaves(gn)):
    rel = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
    any_grad_changed |= not np.array_equal(a, b)
    good = rel < _GRAD_REL_MAX and bool(np.isfinite(b).all())
    ok &= good
    lines.append(f"  [{'ok' if good else 'FAIL'}] grad {name}: rel L2 {rel:.4%}")
  ok &= any_grad_changed  # the key must reach the backward (fp8 cotangent or fp8 forward activations)
  print(f"{label}: {'PASS' if ok else 'FAIL'}\n" + "\n".join(lines), flush=True)
  return ok


def _layer_checks():
  results = {}
  base = {"moe_chunk_pipeline": True, "moe_router_grad_from_wo_input": True}
  cfg_off = _cfg(**base)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_off), cfg_off.mesh_axes)
  batch = int(cfg_off.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_off.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_off, mesh), nnx.Param, ...)
  ref, text_off = _run(cfg_off, mesh, params, x)
  ref2, _ = _run(cfg_off, mesh, params, x)
  results["L0_off_deterministic"] = _bit_identical(ref, ref2)
  print(f"L0 keys off twice bit-identical: {results['L0_off_deterministic']}", flush=True)

  new, text_on = _run(_cfg(**base, moe_fp8_combine=True), mesh, params, x)
  results["L1_fp8_combine"] = _close_report("L1 moe_fp8_combine", ref, new, out_should_change=True)
  a2a_off, a2a_on = text_off.count("all_to_all"), text_on.count("all_to_all")
  f8_on = len(re.findall(r"f8_e4m3fn|float8_e4m3fn", text_on))
  results["L1_structure"] = a2a_on > a2a_off and f8_on > 0 and len(re.findall(r"f8_e4m3fn|float8_e4m3fn", text_off)) == 0
  print(f"L1 structure: all_to_all off/on {a2a_off}/{a2a_on}, e4m3 mentions on {f8_on} -> {results['L1_structure']}")

  new, _ = _run(_cfg(**base, moe_fp8_bwd_dispatch=True), mesh, params, x)
  results["L2_fp8_bwd_dispatch"] = _close_report("L2 moe_fp8_bwd_dispatch", ref, new, out_should_change=False)

  new, _ = _run(_cfg(**base, moe_fp8_combine=True, moe_fp8_bwd_dispatch=True), mesh, params, x)
  results["L2b_fp8_both"] = _close_report("L2b moe_fp8_combine + moe_fp8_bwd_dispatch", ref, new, out_should_change=True)

  with mock.patch.object(moe, "ep_combine_reduce_scatter", _broken_no_scale):
    new, _ = _run(_cfg(**base, moe_fp8_combine=True), mesh, params, x)
  med_b = _stats(_row_rel(np.asarray(ref[1], np.float32), np.asarray(new[1], np.float32)))[0]
  results["L3_broken_combine"] = med_b > _BROKEN_MIN
  print(f"L3 broken combine (scale dropped): out median {med_b:.4%} -> {'detected' if med_b > _BROKEN_MIN else 'NOT'}")

  # 3. moe_combine_per_chunk: plain chunk loop (the pipelined loop's tie 3 already orders the chunks).
  plain = {"moe_chunk_pipeline": False, "moe_router_grad_from_wo_input": True}
  ref_p, text_p_off = _run(_cfg(**plain), mesh, params, x)
  new_p, text_p_on = _run(_cfg(**plain, moe_combine_per_chunk=True), mesh, params, x)
  ties_off, ties_on = text_p_off.count("moe_combine_order"), text_p_on.count("moe_combine_order")
  same = _bit_identical(ref_p, new_p)
  results["C1_per_chunk_plain"] = same and ties_off == 0 and ties_on >= cfg_off.num_moe_token_chunks - 1
  print(f"C1 moe_combine_per_chunk (plain loop): bit-identical {same}, ties off/on {ties_off}/{ties_on} -> "
        f"{results['C1_per_chunk_plain']}")
  _, text_pp_on = _run(_cfg(**base, moe_combine_per_chunk=True), mesh, params, x)
  results["C2_per_chunk_pipelined_noop"] = text_pp_on.count("moe_combine_order") == 0
  print(f"C2 moe_combine_per_chunk (pipelined loop) ties: {text_pp_on.count('moe_combine_order')} (want 0, no-op)")
  # Positive control for C1: a tie that is NOT value-preserving must break bit-identity.
  with mock.patch.object(moe, "_order_after", lambda v, d: v + (jnp.reshape(d, (-1,))[0] * 1e-2).astype(v.dtype)):
    bad_p, _ = _run(_cfg(**plain, moe_combine_per_chunk=True), mesh, params, x)
  results["C3_broken_tie_detected"] = not _bit_identical(ref_p, bad_p)
  print(f"C3 broken tie (adds 1e-2 * dep): detected {results['C3_broken_tie_detected']}")
  return results


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


class ConfigTest(absltest.TestCase):
  """The validator of the new keys."""

  def _init(self, **kw):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="moe_fp8_combine_config",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        **kw,
    )

  def test_defaults_off(self):
    cfg = self._init()
    self.assertFalse(cfg.moe_fp8_combine)
    self.assertFalse(cfg.moe_fp8_bwd_dispatch)
    self.assertFalse(cfg.moe_fp8_combine_pin_sparse_core)
    self.assertFalse(cfg.moe_combine_per_chunk)
    self.assertEqual(cfg.moe_fp8_bwd_dispatch_qtype, "float8_e4m3fn")

  def test_requires_ring_of_experts(self):
    with self.assertRaisesRegex(ValueError, "moe_fp8_combine=True requires use_ring_of_experts=True"):
      self._init(moe_fp8_combine=True)
    with self.assertRaisesRegex(ValueError, "moe_fp8_bwd_dispatch=True requires use_ring_of_experts=True"):
      self._init(moe_fp8_bwd_dispatch=True)

  def test_rejects_bad_combinations(self):
    # The validator itself, on stand-in settings: a one-device test config cannot hold use_ring_of_experts=True.
    def check(**kw):
      settings = {
          "use_ring_of_experts": True,
          "moe_fp8_combine": False,
          "moe_fp8_bwd_dispatch": False,
          "moe_fp8_bwd_dispatch_qtype": "float8_e4m3fn",
          "moe_fp8_combine_pin_sparse_core": False,
          "moe_quantize_combine_bwd_method": "",
          "moe_combine_bwd_direct_qarray": False,
      }
      settings.update(kw)
      types.MaxTextConfig.validate_moe_fp8_combine(SimpleNamespace(**settings))

    check(moe_fp8_combine=True, moe_fp8_bwd_dispatch=True)  # the bench settings pass
    check(moe_fp8_bwd_dispatch=True, moe_fp8_bwd_dispatch_qtype="float8_e5m2")
    with self.assertRaisesRegex(ValueError, "moe_fp8_bwd_dispatch_qtype must be"):
      check(moe_fp8_bwd_dispatch=True, moe_fp8_bwd_dispatch_qtype="int8")
    with self.assertRaisesRegex(ValueError, "has no effect without moe_fp8_combine=True"):
      check(moe_fp8_combine_pin_sparse_core=True)
    with self.assertRaisesRegex(ValueError, "unset moe_quantize_combine_bwd_method"):
      check(moe_fp8_combine=True, moe_quantize_combine_bwd_method="rowwise")
    with self.assertRaisesRegex(ValueError, "moe_combine_bwd_direct_qarray=True"):
      check(moe_fp8_bwd_dispatch=True, moe_combine_bwd_direct_qarray=True)
    with self.assertRaisesRegex(ValueError, "requires use_ring_of_experts=True"):
      check(use_ring_of_experts=False, moe_fp8_combine=True)


class HelperTest(absltest.TestCase):
  """quantize_rows_fp8 / dequantize_rows_fp8 and _order_after on one device."""

  def test_roundtrip_error_and_zero_rows(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (512, 256), jnp.float32)
    x = x.at[3].set(0.0)
    q, s = moe.quantize_rows_fp8(x, jnp.float8_e4m3fn)
    self.assertEqual(q.dtype, jnp.float8_e4m3fn)
    self.assertEqual(s.shape, (512, 1))
    y = moe.dequantize_rows_fp8(q, s, jnp.float32)
    np.testing.assert_array_equal(y[3], np.zeros(256, np.float32))  # an all-zero row stays exactly zero
    r = _row_rel(np.delete(np.asarray(x), 3, 0), np.delete(np.asarray(y), 3, 0))
    self.assertTrue(_in(float(np.median(r)), _E4M3_MEDIAN_BAND), float(np.median(r)))
    # The row max maps to the e4m3 max exactly, so it survives the round trip to f32 rounding of the scale.
    np.testing.assert_allclose(np.max(np.abs(y), -1), np.max(np.abs(np.asarray(x)), -1), rtol=1e-6)

  def test_order_after_is_value_preserving(self):
    v = jax.random.normal(jax.random.PRNGKey(1), (64,), jnp.float32)
    d = jax.random.normal(jax.random.PRNGKey(2), (4, 8), jnp.bfloat16)
    np.testing.assert_array_equal(moe._order_after(v, d), v)  # pylint: disable=protected-access
    gv, gd = jax.grad(lambda a, b: jnp.sum(moe._order_after(a, b) * 3.0), argnums=(0, 1))(v, d)  # pylint: disable=protected-access
    np.testing.assert_array_equal(gv, jnp.full_like(v, 3.0))
    np.testing.assert_array_equal(gd, jnp.zeros_like(d))

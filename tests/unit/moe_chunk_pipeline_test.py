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
"""The software-pipelined chunk loop (`moe_chunk_pipeline=True`) against the plain chunk loop, on a CPU mesh.

The pipelined loop runs the same chunks through the same ops and only adds optimization
barriers (`moe.barrier_tie`) that fix the order the chunks' collectives are issued in.
So it must reproduce the plain loop exactly, alone and together with
`ring_of_experts_row_major_reduce_scatter=True`.

Runs as a subprocess so the forced device count takes effect before JAX initializes.
The mesh is 8 CPU devices, fsdp1 x context4 x expert2 under cp-as-ep, so the
expert-parallel group is 8, as in the Qwen3.5-397B configuration
(fsdp32 x context4 x expert2). The layer is a shrunk Qwen3.5 routed MoE (16 experts,
top 4) with ragged sort, a 2x ragged buffer, local routing and two token chunks, as in
that configuration. The GMM is the megablox kernel in interpret mode, since tokamax
gmm v2 does not run on CPU.

XLA:CPU keeps the barriers, and with its default excess precision a barrier changes where
bfloat16 intermediates get rounded (the output then differs by ~5e-4 and the loss by ~4e-5
relative). The child therefore runs with --xla_allow_excess_precision=false, so every
bfloat16 value is rounded where the program says, in both loops. That is what TPU does
here: every tensor a tie holds is a kernel or collective output, already rounded to
bfloat16 in memory. So this checks the wiring (every chunk gets its own tokens, routing
and gradients), not TPU numerics, and a TPU run should still be held to its loss band.

Pre-registered comparison, per case: the forward output must be bit-identical, and the
loss and every gradient (router kernel, the three expert kernels, the input) must agree
to reduction order, max |new - old| <= 1e-5 * max |old|. Whether they are also
bit-identical is reported. Cases: random, real (softmax + top-k) and forced routing
(padded and duplicate slots), a nonzero load-balance loss, a dropless buffer, the router
in bfloat16, the non-local (all-gather-then-route) routing path, four chunks, one chunk,
expert parallelism off (fsdp8, where the switch must do nothing), the expert-weight
prefetch barrier on in both loops, and the row-major reduce-scatter layout with the
pipeline. (`moe_roe_row_major_reduce_scatter_test.py` covers the layout option alone.)

Two-sided checks:
  - Structure: the pipelined layer must trace exactly 4 x (chunks - 1) barrier ties in one
    forward (next chunk's token gather, this chunk's combine, next chunk's combine, this
    chunk's reduce-scatter) and the plain layer none, so a switch that silently fell back
    to the plain loop cannot pass. With the layout switch on there must be one layout
    constraint per chunk, and each must request the row-major layout.
  - Every output of every barrier must be read. XLA drops a barrier output that nothing
    reads, together with the dependency it carries, so an unread output is a tie that
    silently orders nothing. Checked on the forward and on the gradient jaxpr.
  - Positive controls that must FAIL: a tie that zeroes one token row of the tensor it
    orders (numerics), one that blocks the gradient of that tensor (gradients only; the
    forward is bit-identical), one that leaves the barrier's second output unread
    (structure only; the numerics are bit-identical), and one that does not tie at all.
  - No extra work in the backward. The decoder rematerializes each layer's forward in the
    backward pass. If a value the backward reads came to depend on a chunk's reduce-scatter
    (say, a tie made of a custom call, or of arithmetic on its dependency), that
    recomputation would re-run the reduce-scatter and the gather-reduce that feeds it, work
    the plain loop does not do. With barriers, JAX does recompute the reduce-scatter that
    tie 3 depends on, as an operand of the recomputed barrier, but nothing in the backward
    reads that barrier output and XLA removes it. The check compiles
    jax.grad(jax.checkpoint(loss)) for the plain and the pipelined loop, with the loss linear
    in the layer output as behind the decoder's residual add, and requires the compiled CPU
    programs to have the same collectives. Its positive control is a tie that makes the value
    it holds depend on the value of its dependency: the numbers are bit-identical, the
    structure check passes, and the compiled backward has one extra reduce-scatter.
  - With --xla_tpu_aggressive_opt_barrier_removal=true or ENABLED in effect the layer must
    run the plain loop and warn, and compile_xla_flags must override LIBTPU_INIT_ARGS.
"""

import collections
import os
import re
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.extend import core as jex_core
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "MOE_CHUNK_PIPELINE_CHECKS_PASSED"
_REL_TOL = 1e-5
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
# The unpatched tie: the controls below wrap it, and `mock.patch` must not make them call themselves.
_BARRIER_TIE = moe.barrier_tie


@pytest.mark.cpu_only
def test_chunk_pipeline_matches_plain_chunk_loop_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {_XLA_FLAGS}"
  env["JAX_PLATFORMS"] = "cpu"
  # The child runs this file as a script, so `tests.utils` is importable only with the repo root on the path.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _cfg(pipeline, **overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh, with the pipelined loop on or off."""
  kwargs = {
      "run_name": f"moe_chunk_pipeline_{pipeline}",
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
      # Tokamax gmm v2 is TPU-only; both loops share whichever GMM runs, so CPU uses megablox.
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
      "moe_chunk_pipeline": pipeline,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _build(cfg, mesh, force_pipeline=False):
  """The routed MoE layer for `cfg`; `force_pipeline` turns the pipelined loop on for this layer only."""
  model = moe.RoutedMoE(
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
  if force_pipeline:
    # The layer reads this attribute before the config; the config validator rejects one chunk.
    model.moe_chunk_pipeline = True
  return model


def _loss_fn(graphdef, rest):
  def loss_fn(params, x, forced):
    model = nnx.merge(graphdef, params, rest)
    out, lb_loss, _ = model(x, forced_routed_experts=forced)
    loss = jnp.mean(out.astype(jnp.float32) ** 2)
    if lb_loss is not None:
      loss = loss + lb_loss.astype(jnp.float32)
    return loss, out

  return loss_fn


def _subjaxprs(eqn):
  """The jaxprs nested in an equation's parameters."""
  for param in eqn.params.values():
    for sub in param if isinstance(param, (list, tuple)) else (param,):
      # A closed jaxpr wraps the jaxpr; some JAX versions make them one class.
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      if hasattr(sub, "eqns"):
        yield sub


def _count(jaxpr, name):
  """Number of `name` equations in a jaxpr, including nested jaxprs."""
  return sum((eqn.primitive.name == name) + sum(_count(sub, name) for sub in _subjaxprs(eqn)) for eqn in jaxpr.eqns)


def _layouts(jaxpr):
  """(requested major_to_minor, operand rank) of every layout_constraint equation, including nested jaxprs."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "layout_constraint":
      found.append((tuple(eqn.params["layout"].major_to_minor), eqn.invars[0].aval.ndim))
    for sub in _subjaxprs(eqn):
      found += _layouts(sub)
  return found


def _is_barrier(eqn):
  """An optimization_barrier equation, or a custom_vjp call whose body is a single one (the forward-only tie)."""
  if eqn.primitive.name == "optimization_barrier":
    return True
  if not eqn.primitive.name.startswith("custom_vjp_call"):
    return False
  body = next(_subjaxprs(eqn), None)
  return body is not None and len(body.eqns) == 1 and body.eqns[0].primitive.name == "optimization_barrier"


def _unread_barrier_outputs(jaxpr):
  """Barrier outputs that nothing reads, as 'barrier k output i' strings, over all nested jaxprs.

  An output counts as read if an equation in the same jaxpr takes it as an operand or the
  jaxpr returns it. A barrier inside a forward-only tie's custom_vjp body is checked at the call.
  """
  unread = []
  read = {v for eqn in jaxpr.eqns for v in eqn.invars if not isinstance(v, jex_core.Literal)}
  read |= {v for v in jaxpr.outvars if not isinstance(v, jex_core.Literal)}
  for k, eqn in enumerate(jaxpr.eqns):
    if _is_barrier(eqn):
      unread += [f"barrier {k} output {i}" for i, v in enumerate(eqn.outvars) if v not in read]
    else:
      for sub in _subjaxprs(eqn):
        unread += _unread_barrier_outputs(sub)
  return unread


def _run(cfg, mesh, params, x, forced, force_pipeline=False):
  """Loss, output and gradients for `cfg`, and the traced structure of one forward and of the gradient."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh, force_pipeline), nnx.Param, ...)
    loss_fn = _loss_fn(graphdef, rest)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    forward = jax.make_jaxpr(loss_fn)(params, x, forced).jaxpr
    gradient = jax.make_jaxpr(grad_fn)(params, x, forced).jaxpr
    layouts = _layouts(forward)
    structure = {
        "barriers": _count(forward, "optimization_barrier"),
        "layout_constraints": len(layouts),
        "not_row_major": [m for m, ndim in layouts if m != tuple(range(ndim))],
        "unread": _unread_barrier_outputs(forward) + _unread_barrier_outputs(gradient),
    }
    (loss, out), grads = jax.jit(grad_fn)(params, x, forced)
  return jax.device_get((loss, out, grads)), structure


def _compare(label, old, new):
  """Prints the per-leaf report; returns (passed, bit_identical)."""
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  out_old, out_new = np.asarray(out_old).astype(np.float32), np.asarray(out_new).astype(np.float32)
  out_equal = bool(np.array_equal(out_old, out_new))
  passed, identical = out_equal, out_equal
  lines = [f"  [{'ok' if out_equal else 'FAIL'}] output bit-identical: max|d|={np.max(np.abs(out_new - out_old)):.3e}"]
  leaves_old = jax.tree_util.tree_flatten_with_path(grads_old)[0]
  leaves_new = jax.tree_util.tree_leaves(grads_new)
  named = [("loss", loss_old, loss_new)] + [
      (jax.tree_util.keystr(path), a, b) for (path, a), b in zip(leaves_old, leaves_new)
  ]
  for name, a, b in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    diff, scale = float(np.max(np.abs(b - a))), float(np.max(np.abs(a)))
    same = bool(np.array_equal(a, b))
    # Every comparison is False on NaN, so a NaN fails.
    ok = diff <= _REL_TOL * scale and bool(np.isfinite(b).all()) and scale > 0
    passed &= ok
    identical &= same
    lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: max|d|={diff:.3e} max|old|={scale:.3e} bit-identical={same}")
  print(f"{label}: {'PASS' if passed else 'FAIL'} (bit-identical: {identical})", flush=True)
  print("\n".join(lines), flush=True)
  return passed, identical


def _inputs(cfg, forced_mode, bf16_router=False):
  """The layer input and, if `forced_mode`, forced expert ids with padded and duplicate slots."""
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  x = x.astype(jnp.bfloat16) if bf16_router else x
  if not forced_mode:
    return x, None
  keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
  ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
  pos = jnp.arange(_SEQ)[None, :, None]
  ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
  ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
  return x, ids.astype(jnp.int32)


def _case(label, overrides, forced_mode=False, tie=None, new_overrides=None, pipeline=True, force_pipeline=False):
  """Returns (comparison passed, structure ok, bit-identical).

  The plain loop runs `overrides`; the new one runs `overrides` + `new_overrides` with the pipeline
  switch at `pipeline` (or forced on the layer with `force_pipeline`). `tie` replaces `moe.barrier_tie`.
  """
  bf16_router = overrides.get("float32_gate_logits") is False
  cfg_old = _cfg(False, **overrides)
  cfg_new = _cfg(pipeline, **{**overrides, **(new_overrides or {})})
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, forced = _inputs(cfg_old, forced_mode, bf16_router)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, s_old = _run(cfg_old, mesh, params, x, forced)
  with mock.patch.object(moe, "barrier_tie", tie or _BARRIER_TIE):
    new, s_new = _run(cfg_new, mesh, params, x, forced, force_pipeline)
  n = cfg_new.num_moe_token_chunks
  pipelined = (pipeline or force_pipeline) and n > 1
  prefetch = 1 if cfg_new.moe_expert_weight_prefetch is not None else 0
  want = {
      "barriers": (prefetch, prefetch + (4 * (n - 1) if pipelined else 0)),
      "layout_constraints": (
          cfg_old.num_moe_token_chunks if cfg_old.ring_of_experts_row_major_reduce_scatter else 0,
          n if cfg_new.ring_of_experts_row_major_reduce_scatter else 0,
      ),
  }
  structure = all((s_old[k], s_new[k]) == v for k, v in want.items())
  structure &= not any(s[k] for s in (s_old, s_new) for k in ("unread", "not_row_major"))
  print(
      f"{label}: [{'ok' if structure else 'FAIL'}] barriers plain/new={s_old['barriers']}/{s_new['barriers']}"
      f" (want {want['barriers']}), layout constraints {s_old['layout_constraints']}/{s_new['layout_constraints']}"
      f" (want {want['layout_constraints']}, all row-major; not row-major: {s_old['not_row_major']}"
      f" / {s_new['not_row_major']}), unread barrier outputs plain={s_old['unread']} new={s_new['unread']}",
      flush=True,
  )
  passed, identical = _compare(label, old, new)
  return passed, structure, identical


def _zero_one_token(x, dep, backward=False):
  """Deliberately wrong: ties `x` but also zeroes its first token row."""
  x, dep = _BARRIER_TIE(x, dep, backward)
  return x.at[(0,) * (x.ndim - 1)].set(0), dep


def _block_gradient(x, dep, backward=False):
  """Deliberately wrong: ties `x` but drops its gradient."""
  x, dep = _BARRIER_TIE(x, dep, backward)
  return jax.lax.stop_gradient(x), dep


def _unread_dependency_output(x, dep, backward=False):
  """Deliberately wrong: ties `x` but hands back the untied `dep`, so the barrier's second output is unread."""
  x, _ = _BARRIER_TIE(x, dep, backward)
  return x, dep


def _no_tie(x, dep, backward=False):
  """Deliberately wrong: no barrier at all, i.e. a pipeline that silently orders nothing."""
  del backward
  return x, dep


def _sticky_dependency(x, dep, backward=False):
  """Deliberately wrong: ties `x` and also makes its value depend on the value of `dep`, without changing it."""
  x, dep = _BARRIER_TIE(x, dep, backward)
  first = jax.lax.stop_gradient(jnp.ravel(jax.tree_util.tree_leaves(dep)[0])[0])
  return jax.tree_util.tree_map(lambda v: jnp.where(first == jnp.inf, v * 2, v), x), dep


_COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")


def _split_shape(rest):
  """Splits 'shape opcode(...)' at the end of the result shape, which for a tuple is its closing parenthesis."""
  if not rest.startswith("("):
    shape, _, rest = rest.partition(" ")
    return shape, rest
  depth = 0
  for i, char in enumerate(rest):
    depth += (char == "(") - (char == ")")
    if depth == 0:
      return rest[: i + 1], rest[i + 1 :].lstrip()
  return rest, ""


def _collectives(hlo_text):
  """Multiset of (opcode, result shape without layout) of the collectives in a compiled HLO module."""
  found = collections.Counter()
  for line in hlo_text.splitlines():
    if " = " not in line:
      continue
    shape, rest = _split_shape(line.split(" = ", 1)[1])
    op = rest.split("(", 1)[0].removesuffix("-start")
    if op in _COLLECTIVES:
      found[(op, re.sub(r"\{[^}]*\}", "", shape))] += 1
  return found


def _remat_backward(overrides, pipeline, tie=None):
  """Collectives of the compiled jax.grad(jax.checkpoint(loss)), and the reduce_scatter equations in its jaxpr."""
  cfg = _cfg(pipeline, **overrides)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  x, _ = _inputs(cfg, False)
  with (
      mock.patch.object(moe, "barrier_tie", tie or _BARRIER_TIE),
      jax.set_mesh(mesh),
      nn_partitioning.axis_rules(cfg.logical_axis_rules),
  ):
    graphdef, params, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      # Linear in the layer output, as behind the decoder's residual add: the backward does not need the output.
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x)
      loss = jnp.sum(out.astype(jnp.float32) * jax.random.normal(jax.random.PRNGKey(3), out.shape, jnp.float32))
      return loss if lb_loss is None else loss + lb_loss.astype(jnp.float32)

    grad_fn = jax.grad(jax.checkpoint(loss_fn, policy=jax.checkpoint_policies.nothing_saveable), argnums=(0, 1))
    jaxpr = jax.make_jaxpr(grad_fn)(params, x).jaxpr
    hlo = jax.jit(grad_fn).lower(params, x).compile().as_text()
  return _collectives(hlo), _count(jaxpr, "reduce_scatter")


def _remat_backward_check():
  """Returns (the pipelined loop's rematerialized backward has the plain loop's collectives, the control is caught)."""
  row = {"ring_of_experts_row_major_reduce_scatter": True}
  arms = {
      "plain": _remat_backward({}, False),
      "pipelined": _remat_backward(row, True),
      "control_sticky_dependency": _remat_backward(row, True, tie=_sticky_dependency),
  }
  for label, (found, jaxpr_rs) in arms.items():
    output_rs = sum(n for (op, shape), n in found.items() if op == "reduce-scatter" and shape.endswith(f",{_EMB}]"))
    print(
        f"remat backward, {label}: compiled reduce-scatters of the layer output's shape={output_rs};"
        f" reduce_scatter equations in the gradient jaxpr={jaxpr_rs}; collectives={dict(sorted(found.items()))}",
        flush=True,
    )
  same = arms["pipelined"][0] == arms["plain"][0]
  caught = arms["control_sticky_dependency"][0] != arms["plain"][0]
  print(f"remat backward: [{'ok' if same else 'FAIL'}] pipelined == plain; control caught: {caught}", flush=True)
  return same, caught


def _barrier_removal_check():
  """The layer runs the plain loop and warns when the removal flag applies; compile_xla_flags overrides the
  environment."""
  flag = "--xla_tpu_aggressive_opt_barrier_removal"
  cases = (  # label, compile_xla_flags, LIBTPU_INIT_ARGS, whether the pipeline runs
      ("flag unset", "", "", True),
      ("compile_xla_flags true", f"{flag}=true", "", False),
      ("LIBTPU_INIT_ARGS true", "", f"{flag}=true", False),
      ("LIBTPU_INIT_ARGS ENABLED", "", f"{flag}=ENABLED", False),
      ("LIBTPU_INIT_ARGS true, compile_xla_flags false", f"{flag}=false", f"{flag}=true", True),
      ("LIBTPU_INIT_ARGS false, compile_xla_flags true", f"{flag}=true", f"{flag}=false", False),
  )
  ok = True
  for label, compile_flags, env_flags, runs in cases:
    cfg = _cfg(True, use_random_routing=True, compile_xla_flags=compile_flags)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x, forced = _inputs(cfg, False)
    with (
        mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": env_flags}),
        mock.patch.object(moe.max_logging, "warning") as warning,
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg.logical_axis_rules),
    ):
      graphdef, params, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)
      barriers = _count(jax.make_jaxpr(_loss_fn(graphdef, rest))(params, x, forced).jaxpr, "optimization_barrier")
    warned = any("moe_chunk_pipeline" in str(call) for call in warning.call_args_list)
    want = 4 * (cfg.num_moe_token_chunks - 1) if runs else 0
    case_ok = barriers == want and warned == (not runs)
    ok &= case_ok
    print(
        f"barrier removal, {label}: [{'ok' if case_ok else 'FAIL'}] barriers={barriers} warned={warned}"
        f" expected barriers={want}",
        flush=True,
    )
  return ok


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  assert "--xla_allow_excess_precision=false" in os.environ.get("XLA_FLAGS", ""), f"run with XLA_FLAGS='{_XLA_FLAGS}'"
  rnd = {"use_random_routing": True}
  row = {"ring_of_experts_row_major_reduce_scatter": True}
  # Expert parallelism off: the config then requires no ring of experts, no ragged sort, no random routing and a
  # dropless buffer, and chunks > 1 need the ring. The switch is forced on the layer and must do nothing.
  ep_off = {
      "ici_fsdp_parallelism": 8,
      "ici_context_parallelism": 1,
      "ici_expert_parallelism": 1,
      "per_device_batch_size": 1,
      "use_ring_of_experts": False,
      "use_ragged_sort": False,
      "ragged_buffer_factor": -1.0,
      "num_moe_token_chunks": 1,
      "ring_of_experts_local_routing": False,
  }
  cases = {
      "random": _case("random", rnd),
      "real": _case("real", {}),
      "forced": _case("forced", {}, forced_mode=True),
      "load_balance_loss": _case("load_balance_loss", {"load_balance_loss_weight": 0.01}),
      "random_dropless": _case("random_dropless", {**rnd, "ragged_buffer_factor": -1.0}),
      "real_bf16_router": _case("real_bf16_router", {"float32_gate_logits": False}),
      "random_gathered_routing": _case("random_gathered_routing", {**rnd, "ring_of_experts_local_routing": False}),
      "random_4_chunks": _case("random_4_chunks", {**rnd, "num_moe_token_chunks": 4}),
      "forced_4_chunks": _case("forced_4_chunks", {"num_moe_token_chunks": 4}, forced_mode=True),
      "random_1_chunk": _case("random_1_chunk", {**rnd, "num_moe_token_chunks": 1}, pipeline=False, force_pipeline=True),
      "real_ep_off": _case("real_ep_off", ep_off, pipeline=False, force_pipeline=True),
      "real_with_weight_prefetch": _case("real_with_weight_prefetch", {"moe_expert_weight_prefetch": "forward"}),
      "real_row_major_rs_pipelined": _case("real_row_major_rs_pipelined", {}, new_overrides=row),
  }
  # Each control must fail exactly the check named: 'numerics' (the comparison) or 'structure'.
  controls = {
      "control_zero_one_token": (_case("control_zero_one_token", rnd, tie=_zero_one_token), "numerics"),
      "control_block_gradient": (_case("control_block_gradient", {}, tie=_block_gradient), "numerics"),
      "control_unread_output": (_case("control_unread_output", rnd, tie=_unread_dependency_output), "structure"),
      "control_no_tie": (_case("control_no_tie", rnd, tie=_no_tie), "structure"),
  }
  # The sticky-dependency control passes the checks above (same numbers, same structure); only the rematerialized
  # backward check below can see it.
  sticky = _case("control_sticky_dependency_other_checks", {}, tie=_sticky_dependency, new_overrides=row)
  remat_same, remat_control_caught = _remat_backward_check()
  barrier_removal = _barrier_removal_check()
  summary = {k: {"passed": v[0], "structure": v[1], "bit_identical": v[2]} for k, v in cases.items()}
  summary.update(
      {k: {"passed": v[0], "structure": v[1], "bit_identical": v[2], "must_fail": m} for k, (v, m) in controls.items()}
  )
  summary.update({"remat_backward": remat_same, "barrier_removal": barrier_removal})
  summary["control_sticky_dependency"] = {"passes_other_checks": sticky[0] and sticky[1], "caught": remat_control_caught}
  print("SUMMARY", summary, flush=True)
  failed = [k for k, v in cases.items() if not (v[0] and v[1])]
  failed += [k for k, v in (("remat_backward", remat_same), ("barrier_removal", barrier_removal)) if not v]
  undetected = [k for k, (v, m) in controls.items() if (v[0] if m == "numerics" else v[1])]
  undetected += [] if remat_control_caught else ["control_sticky_dependency"]
  # A sticky control that already fails another check would not show that the backward check is needed.
  failed += [] if sticky[0] and sticky[1] else ["control_sticky_dependency_other_checks"]
  if failed or undetected:
    print(f"cases that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()


class MoeChunkPipelineConfigTest(absltest.TestCase):

  def _init(self, **kwargs):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="moe_chunk_pipeline_config",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        **kwargs,
    )

  def test_requires_chunks_and_ring_of_experts(self):
    with self.assertRaisesRegex(ValueError, "moe_chunk_pipeline=True requires num_moe_token_chunks > 1"):
      self._init(moe_chunk_pipeline=True, use_ring_of_experts=True)

  def test_rejects_barrier(self):
    with self.assertRaisesRegex(ValueError, "moe_chunk_barrier=True order the chunks in opposite ways"):
      self._init(moe_chunk_pipeline=True, moe_chunk_barrier=True, use_ring_of_experts=True, num_moe_token_chunks=2)

  def test_rejects_embedding_chunks(self):
    with self.assertRaisesRegex(ValueError, "does not support num_moe_emb_chunks > 0"):
      self._init(moe_chunk_pipeline=True, use_ring_of_experts=True, num_moe_token_chunks=2, num_moe_emb_chunks=2)


class BarrierTieTest(absltest.TestCase):
  """`moe.barrier_tie` returns both inputs unchanged; its forward-only form leaves the cotangents alone."""

  def test_values_and_cotangents(self):
    x, dep = jnp.arange(6.0).reshape(2, 3), jnp.arange(4.0)
    for backward in (False, True):
      x2, dep2 = moe.barrier_tie(x, dep, backward)
      np.testing.assert_array_equal(x2, x)
      np.testing.assert_array_equal(dep2, dep)
      gx, gdep = jax.grad(lambda a, b, bw=backward: jnp.sum(moe.barrier_tie(a, b, bw)[0] * 3.0), argnums=(0, 1))(x, dep)
      np.testing.assert_array_equal(gx, jnp.full_like(x, 3.0))
      np.testing.assert_array_equal(gdep, jnp.zeros_like(dep))

  def test_unread_output_detector_is_two_sided(self):
    def tied(a, b):
      a, b = moe.barrier_tie(a, b)
      return a + 1.0, b * 2.0

    def half_read(a, b):
      a, _ = moe.barrier_tie(a, b)
      return a + 1.0, b * 2.0

    x = jnp.ones((4,))
    self.assertEqual(_unread_barrier_outputs(jax.make_jaxpr(tied)(x, x).jaxpr), [])
    self.assertEqual(_unread_barrier_outputs(jax.make_jaxpr(half_read)(x, x).jaxpr), ["barrier 0 output 1"])

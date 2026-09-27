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
"""`moe_expert_weight_prefetch` against the default MoE path, on a CPU mesh.

The switch ties the FSDP-gathered expert weights to the routed tokens and the router
logits with an optimization barrier. The barrier is the identity, so the layer must
compute the same output, loss and gradients with the switch on as with it off.

Runs as a subprocess so the forced device count takes effect before JAX initializes.
The mesh is 8 CPU devices, fsdp2 x context2 x expert2 under cp-as-ep, so the expert
weights are FSDP-sharded (and gathered per layer) as in the Qwen3.5-397B configuration
(fsdp32 x context4 x expert2). The layer is a shrunk Qwen3.5 routed MoE (16 experts,
top 4) with float32 master weights, bfloat16 compute, ragged sort, a 2x ragged buffer
and two token chunks, as in that configuration. The GMM is the megablox kernel in interpret mode,
since tokamax gmm v2 does not run on CPU.

Each case compares "forward" and "forward_backward" against "none" with shared
parameters and inputs: the forward output must be bit-identical, and the loss and every
gradient (router kernel, the three expert kernels, the input) must agree to reduction
order, max |new - old| <= 1e-5 * max |old|. The report also says whether each leaf is
bit-identical.

Checks that make the comparison two-sided:
  - Structure: "none" must emit no optimization barrier, "forward" exactly one per layer
    call in the forward and none in the backward, "forward_backward" one in each. The
    barrier must hold the tokens, the router logits and the three expert weights, the
    weights must come from sharding constraints without the fsdp axis, i.e. it holds the
    gathered weights, not the FSDP shards, and every output of the barrier must be used.
    (The CPU compiler strips optimization barriers, so this is checked on the jaxpr.) The
    checker itself must reject a barrier over FSDP-sharded weights, one that leaves the
    logits untied and one with an unused output.
  - With --xla_tpu_aggressive_opt_barrier_removal=true or ENABLED in effect the layer must
    skip the barrier and warn, and `compile_xla_flags` must override LIBTPU_INIT_ARGS.
  - Deliberately wrong barriers must FAIL the numeric comparison while passing the
    structural checks: one that swaps the two up-projection weights (the output changes),
    and one whose backward drops the expert weight cotangents (the output is unchanged;
    only the gradient check can see it). Both are custom_vjps over one barrier, like the
    real "forward" barrier, so only the comparison can tell them apart from it.
"""

import contextlib
import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_post_train_test_config_path, get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_MLP = 128
_SEQ = 64
_PER_DEVICE_BATCH = 0.5
_PASS_TOKEN = "MOE_EXPERT_WEIGHT_PREFETCH_CHECKS_PASSED"
_REL_TOL = 1e-5


@pytest.mark.cpu_only
def test_expert_weight_prefetch_matches_default_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  # The child imports `tests.utils`, so it needs the repository root on its path, not only `src`.
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _cfg(prefetch, **overrides):
  """The shrunk Qwen3.5 routed-MoE config on an fsdp2 x context2 x expert2 cp-as-ep mesh."""
  kwargs = {
      "run_name": f"moe_expert_weight_prefetch_{prefetch}",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": True,
      "num_experts": _NUM_EXPERTS,
      "num_experts_per_tok": _TOP_K,
      "base_emb_dim": _EMB,
      "base_mlp_dim": _MLP,
      "base_moe_mlp_dim": _MLP,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      # The batch is sharded over fsdp x expert (4 ways), one sequence per shard as in the 397B configuration.
      "per_device_batch_size": _PER_DEVICE_BATCH,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "float32_gate_logits": True,
      "megablox": True,
      "sparse_matmul": True,
      # Tokamax gmm v2 is TPU-only; both paths share whichever GMM runs, so CPU uses megablox.
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "ring_of_experts_local_routing": True,
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": 2,
      "ici_context_parallelism": 2,
      "ici_expert_parallelism": 2,
      "context_parallel_load_balance": False,
      "moe_expert_weight_prefetch": prefetch,
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


def _loss_fn(graphdef, rest):
  def loss_fn(params, x):
    model = nnx.merge(graphdef, params, rest)
    out, lb_loss, _ = model(x)
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


def _count_barriers(jaxpr):
  """Number of optimization_barrier equations in a jaxpr, including nested jaxprs."""
  return sum(
      (eqn.primitive.name == "optimization_barrier") + sum(_count_barriers(sub) for sub in _subjaxprs(eqn))
      for eqn in jaxpr.eqns
  )


def _barrier_operands(jaxpr):
  """For each barrier: the producer spec of each expert-weight operand, whether it holds the tokens and logits,
  and whether every one of its outputs is used.

  A barrier is an `optimization_barrier` equation or a custom_vjp call wrapping one. An expert weight is a
  3-D operand whose leading dimension is the number of experts; the tokens are [batch, seq, emb] and the
  router logits [batch, seq, experts]. A weight's producer is looked up in the jaxpr that holds the barrier;
  anything other than a sharding constraint is reported as its primitive name. An output is used if an
  equation of that jaxpr reads it or the jaxpr returns it.
  """
  found = []
  producers = {v: eqn for eqn in jaxpr.eqns for v in eqn.outvars}
  used = {id(v) for eqn in jaxpr.eqns for v in eqn.invars} | {id(v) for v in jaxpr.outvars}
  for eqn in jaxpr.eqns:
    is_barrier = eqn.primitive.name == "optimization_barrier" or (
        eqn.primitive.name.startswith("custom_vjp_call") and any(_count_barriers(sub) for sub in _subjaxprs(eqn))
    )
    if is_barrier:
      specs, tokens, logits = [], False, False
      for v in eqn.invars:
        shape = tuple(getattr(v.aval, "shape", ()))
        if len(shape) != 3:
          continue
        if shape[0] == _NUM_EXPERTS:
          producer = producers.get(v)
          if producer is not None and producer.primitive.name == "sharding_constraint":
            specs.append(tuple(producer.params["sharding"].spec))
          else:
            specs.append(producer.primitive.name if producer is not None else "input")
        tokens |= shape[1:] == (_SEQ, _EMB)
        logits |= shape[1:] == (_SEQ, _NUM_EXPERTS)
      found.append((specs, tokens, logits, all(id(v) in used for v in eqn.outvars)))
    else:
      for sub in _subjaxprs(eqn):
        found.extend(_barrier_operands(sub))
  return found


def _holds_gathered_weights(found):
  """True if there is a barrier and each one holds the tokens, the router logits, and three expert weights
  constrained to a spec without fsdp (i.e. gathered over FSDP), and every one of its outputs is used."""

  def gathered(spec):
    if not isinstance(spec, tuple):
      return False
    axes = [a for entry in spec for a in (entry if isinstance(entry, tuple) else (entry,))]
    return "fsdp" not in axes

  return bool(found) and all(
      tokens and logits and outputs_used and len(specs) == 3 and all(gathered(spec) for spec in specs)
      for specs, tokens, logits, outputs_used in found
  )


def _run(cfg, mesh, params, x):
  """Loss, output and gradients for `cfg`, its barrier counts, and the specs of the barrier's weights."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)
    loss_fn = _loss_fn(graphdef, rest)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    forward_jaxpr = jax.make_jaxpr(loss_fn)(params, x).jaxpr
    counts = (_count_barriers(forward_jaxpr), _count_barriers(jax.make_jaxpr(grad_fn)(params, x).jaxpr))
    weight_specs = _barrier_operands(forward_jaxpr)
    (loss, out), grads = jax.jit(grad_fn)(params, x)
  return jax.device_get((loss, out, grads)), counts, weight_specs


def _compare(label, old, new):
  """Prints the per-leaf report and returns whether the comparison passed."""
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  out_old, out_new = np.asarray(out_old).astype(np.float32), np.asarray(out_new).astype(np.float32)
  out_equal = bool(np.array_equal(out_old, out_new))
  passed = out_equal
  lines = [f"  [{'ok' if out_equal else 'FAIL'}] output bit-identical: max|d|={np.max(np.abs(out_new - out_old)):.3e}"]
  leaves_old = jax.tree_util.tree_flatten_with_path(grads_old)[0]
  leaves_new = jax.tree_util.tree_leaves(grads_new)
  named = [("loss", loss_old, loss_new)] + [
      (jax.tree_util.keystr(path), a, b) for (path, a), b in zip(leaves_old, leaves_new)
  ]
  for name, a, b in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    diff, scale = float(np.max(np.abs(b - a))), float(np.max(np.abs(a)))
    # Every comparison is False on NaN, so a NaN fails.
    ok = diff <= _REL_TOL * scale and bool(np.isfinite(b).all()) and scale > 0
    passed &= ok
    lines.append(
        f"  [{'ok' if ok else 'FAIL'}] {name}: max|d|={diff:.3e} max|old|={scale:.3e}"
        f" bit-identical={bool(np.array_equal(a, b))}"
    )
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _structure_ok(label, prefetch, counts, weight_specs):
  """Checks the barrier counts and, with the switch on, that the barrier holds the FSDP-gathered weights."""
  expected = {"none": (0, 0), "forward": (1, 1), "forward_backward": (1, 2)}[prefetch]
  ok = counts == expected
  ok = ok and (_holds_gathered_weights(weight_specs) if prefetch != "none" else not weight_specs)
  print(
      f"{label}: [{'ok' if ok else 'FAIL'}] barriers (forward, forward+backward)={counts} expected={expected};"
      f" barrier operands (weight specs, tokens?, logits?, outputs used?)={weight_specs}",
      flush=True,
  )
  return ok


def _checker_self_test(mesh):
  """The structural checker must accept a barrier over gathered weights, tokens and logits whose outputs are all
  used, and reject one over FSDP-sharded weights, one without the logits and one with an unused output."""
  batch = int(_PER_DEVICE_BATCH * _NUM_DEVICES)
  w = jnp.zeros((_NUM_EXPERTS, _EMB, _MLP), jnp.bfloat16)
  x = jnp.zeros((batch, _SEQ, _EMB), jnp.bfloat16)
  logits = jnp.zeros((batch, _SEQ, _NUM_EXPERTS), jnp.float32)

  def program(weight_spec, with_logits=True, use_all_outputs=True):
    def f(x, logits, w):
      w = jax.lax.with_sharding_constraint(w, jax.sharding.NamedSharding(mesh, weight_spec))
      x, logits, (w0, w1, wo) = jax.lax.optimization_barrier((x, logits if with_logits else None, (w, w, w)))
      return (x, logits, w0, w1, wo) if use_all_outputs else (x, logits, w0, w1)

    return _barrier_operands(jax.make_jaxpr(f)(x, logits, w).jaxpr)

  gathered = jax.sharding.PartitionSpec("expert", None, None)
  cases = {
      "sharded weights": (program(jax.sharding.PartitionSpec("expert", "fsdp", None)), False),
      "logits untied": (program(gathered, with_logits=False), False),
      "an unused output": (program(gathered, use_all_outputs=False), False),
      "gathered weights, tokens and logits": (program(gathered), True),
  }
  ok = all(_holds_gathered_weights(found) == want for found, want in cases.values())
  print(f"checker self-test: [{'ok' if ok else 'FAIL'}] {cases}", flush=True)
  return ok


def _case(label, overrides, prefetch, control=None):
  """Runs "none" and `prefetch` on shared parameters; `control` is applied to the `prefetch` run only.

  Returns whether the numeric comparison passed and whether the structural checks passed.
  """
  cfg_old = _cfg("none", **overrides)
  cfg_new = _cfg(prefetch, **overrides)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x = jax.random.normal(jax.random.PRNGKey(7), (int(_PER_DEVICE_BATCH * _NUM_DEVICES), _SEQ, _EMB), jnp.float32)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, counts_old, specs_old = _run(cfg_old, mesh, params, x)
  with control if control is not None else contextlib.nullcontext():
    new, counts_new, specs_new = _run(cfg_new, mesh, params, x)
  structural = _structure_ok(f"{label} [none]", "none", counts_old, specs_old)
  structural &= _structure_ok(f"{label} [{prefetch}]", prefetch, counts_new, specs_new)
  return _compare(label, old, new), structural


def _barrier_removal_check():
  """The layer skips the barrier and warns when the removal flag applies, and compile_xla_flags overrides the
  environment."""
  flag = "--xla_tpu_aggressive_opt_barrier_removal"
  cases = (  # label, compile_xla_flags, LIBTPU_INIT_ARGS, whether the barrier is kept
      ("flag unset", "", "", True),
      ("compile_xla_flags true", f"{flag}=true", "", False),
      ("LIBTPU_INIT_ARGS true", "", f"{flag}=true", False),
      ("LIBTPU_INIT_ARGS ENABLED", "", f"{flag}=ENABLED", False),
      ("LIBTPU_INIT_ARGS true, compile_xla_flags false", f"{flag}=false", f"{flag}=true", True),
      ("LIBTPU_INIT_ARGS false, compile_xla_flags true", f"{flag}=true", f"{flag}=false", False),
  )
  x = jnp.zeros((int(_PER_DEVICE_BATCH * _NUM_DEVICES), _SEQ, _EMB), jnp.float32)
  ok = True
  for label, compile_flags, env_flags, kept in cases:
    cfg = _cfg("forward", use_random_routing=True, compile_xla_flags=compile_flags)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with (
        mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": env_flags}),
        mock.patch.object(moe.max_logging, "warning") as warning,
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg.logical_axis_rules),
    ):
      graphdef, params, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)
      barriers = _count_barriers(jax.make_jaxpr(_loss_fn(graphdef, rest))(params, x).jaxpr)
    warned = any("moe_expert_weight_prefetch" in str(call) for call in warning.call_args_list)
    case_ok = barriers == (1 if kept else 0) and warned == (not kept)
    ok &= case_ok
    print(
        f"barrier removal, {label}: [{'ok' if case_ok else 'FAIL'}] barriers={barriers} warned={warned}"
        f" expected barriers={1 if kept else 0}",
        flush=True,
    )
  return ok


def _swapped_up_projections():
  """A forward-only barrier that returns w0 and w1 swapped: a wiring bug that changes the output.

  Like the real barrier it is a custom_vjp over one optimization barrier, so it passes the structural checks.
  """

  def swap(operands):
    *rest, (w0, w1, wo, b0, b1, bo) = operands
    return (*rest, (w1, w0, wo, b0, b1, bo))

  @jax.custom_vjp
  def wrong(operands):
    return swap(jax.lax.optimization_barrier(operands))

  def wrong_fwd(operands):
    return swap(jax.lax.optimization_barrier(operands)), None

  def wrong_bwd(_, cotangents):
    return (swap(cotangents),)

  wrong.defvjp(wrong_fwd, wrong_bwd)
  return mock.patch.object(moe, "_forward_optimization_barrier", wrong)


def _dropped_weight_cotangents():
  """A forward-only barrier whose backward zeroes the expert-weight cotangents."""

  @jax.custom_vjp
  def wrong(operands):
    return jax.lax.optimization_barrier(operands)

  def wrong_fwd(operands):
    return jax.lax.optimization_barrier(operands), None

  def wrong_bwd(_, cotangents):
    *ct_rest, ct_weights = cotangents
    return ((*ct_rest, jax.tree.map(jnp.zeros_like, ct_weights)),)

  wrong.defvjp(wrong_fwd, wrong_bwd)
  return mock.patch.object(moe, "_forward_optimization_barrier", wrong)


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  random_routing = {"use_random_routing": True}
  cfg = _cfg("none")
  results = {
      "checker_self_test": _checker_self_test(Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)),
      "barrier_removal": _barrier_removal_check(),
      "forward_random": all(_case("forward_random", random_routing, "forward")),
      "forward_real": all(_case("forward_real", {}, "forward")),
      "forward_load_balance_loss": all(_case("forward_load_balance_loss", {"load_balance_loss_weight": 0.01}, "forward")),
      "forward_backward_random": all(_case("forward_backward_random", random_routing, "forward_backward")),
      "forward_backward_real": all(_case("forward_backward_real", {}, "forward_backward")),
  }
  # (numeric comparison passed, structural checks passed) for each deliberately wrong barrier.
  controls = {
      "control_swapped_up_projections": _case(
          "control_swapped_up_projections", random_routing, "forward", control=_swapped_up_projections()
      ),
      "control_dropped_weight_cotangents": _case(
          "control_dropped_weight_cotangents", random_routing, "forward", control=_dropped_weight_cotangents()
      ),
  }
  print("SUMMARY", {**results, **controls}, flush=True)
  failed = [k for k, v in results.items() if not v]
  undetected = [k for k, (numeric, _) in controls.items() if numeric]
  # A control must be caught by the numeric comparison alone, so it has to pass the structural checks.
  structure_failed = [k for k, (_, structural) in controls.items() if not structural]
  if failed or undetected or structure_failed:
    print(
        f"cases that failed: {failed}; positive controls that were NOT detected: {undetected};"
        f" positive controls that failed the structural checks: {structure_failed}",
        flush=True,
    )
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()


class ExpertWeightPrefetchConfigTest(absltest.TestCase):

  def test_rejects_unknown_mode(self):
    with self.assertRaisesRegex(ValueError, "moe_expert_weight_prefetch"):
      pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="moe_expert_weight_prefetch_unknown",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          moe_expert_weight_prefetch="early",
      )

  def test_defaults_to_off(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="moe_expert_weight_prefetch_default",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
    )
    self.assertIsNone(cfg.moe_expert_weight_prefetch)

  def test_rejects_dense_matmul(self):
    with self.assertRaisesRegex(ValueError, "moe_expert_weight_prefetch requires sparse_matmul=True"):
      pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="moe_expert_weight_prefetch_dense",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          sparse_matmul=False,
          moe_expert_weight_prefetch="forward",
      )

  def test_rl_config_rejects_dense_matmul(self):
    """RLConfig does not run MaxTextConfig's validators, so it carries its own copy of the check."""
    rl_args = (["", get_post_train_test_config_path("rl")],)
    rl_kwargs = {"skip_jax_distributed_system": True, "config_class": config_types.RLConfig}
    with self.assertRaisesRegex(ValueError, "moe_expert_weight_prefetch requires sparse_matmul=True"):
      pyconfig.initialize(*rl_args, **rl_kwargs, sparse_matmul=False, moe_expert_weight_prefetch="forward")
    cfg = pyconfig.initialize(*rl_args, **rl_kwargs, sparse_matmul=True, moe_expert_weight_prefetch="forward")
    self.assertEqual(cfg.moe_expert_weight_prefetch, "forward")

  def test_detects_barrier_removal_flag(self):
    flag = "--xla_tpu_aggressive_opt_barrier_removal"
    strips = max_utils.xla_strips_optimization_barriers
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "", "XLA_FLAGS": ""}):
      self.assertTrue(strips(f"--xla_foo=1 {flag}=true"))
      self.assertFalse(strips(f"{flag}=false"))
      # libtpu also takes ENABLED, which strips barriers like true, and DISABLED, which keeps them like false.
      self.assertTrue(strips(f"{flag}=ENABLED"))
      self.assertTrue(strips(f"{flag}=enabled"))
      self.assertFalse(strips(f"{flag}=DISABLED"))
      self.assertFalse(strips("--xla_foo=1"))
      self.assertFalse(strips(""))
      # compile_xla_flags is parsed as the compile parses it, which rejects a repeated flag.
      with self.assertRaisesRegex(ValueError, "Duplicate flag"):
        strips(f"{flag}=true {flag}=false")
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": f"{flag}=true", "XLA_FLAGS": ""}):
      self.assertTrue(strips(""))
      # The per-compile options override the environment.
      self.assertFalse(strips(f"{flag}=false"))
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": f"{flag}=false", "XLA_FLAGS": ""}):
      self.assertTrue(strips(f"{flag}=true"))
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": f"{flag}=true --xla_foo {flag}=false", "XLA_FLAGS": ""}):
      # In the environment the last setting wins, and tokens compile_xla_flags would reject are skipped.
      self.assertFalse(strips(""))
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "", "XLA_FLAGS": f"{flag}=true"}):
      self.assertTrue(strips(""))
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": f"{flag}=ENABLED", "XLA_FLAGS": ""}):
      self.assertTrue(strips(""))
      self.assertFalse(strips(f"{flag}=DISABLED"))

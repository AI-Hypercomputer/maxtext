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
"""`ring_of_experts_row_major_reduce_scatter` against the default ring of experts, on a CPU mesh.

The option only puts a forward-only layout constraint (`moe.forward_row_major`) on the
input of the expert-parallel reduce-scatter. It changes no value, so the layer must
compute the same output, loss and gradients with it on as with it off.

Runs as a subprocess so the forced device count takes effect before JAX initializes.
The mesh is 8 CPU devices, fsdp1 x context4 x expert2 under cp-as-ep, so the
expert-parallel group is 8, as in the Qwen3.5-397B configuration
(fsdp32 x context4 x expert2). The layer is a shrunk Qwen3.5 routed MoE (16 experts,
top 4) with ragged sort, a 2x ragged buffer and local routing, with two token chunks and
with one. The GMM is the megablox kernel in interpret mode, since tokamax gmm v2 does not
run on CPU. The child runs with --xla_allow_excess_precision=false, so bfloat16 values are
rounded where the program says in both runs and an op boundary cannot move a rounding.

Per case: the forward output must be bit-identical, and the loss and every gradient
(router kernel, the three expert kernels, the input) must agree to reduction order,
max |new - old| <= 1e-5 * max |old|. Whether they are also bit-identical is reported.

Two-sided checks:
  - Structure: with the option on, the forward must carry one layout constraint per chunk,
    each requesting the row-major layout (major_to_minor = 0, 1, ..., rank - 1), and with it
    off none. Positive controls that must FAIL it while the numbers stay bit-identical: an
    option that does nothing, and one that requests the reversed (column-major) layout.
  - Forward only: the gradient of `forward_row_major` carries one layout constraint (the
    forward's), while a plain `with_layout_constraint` also constrains the cotangent.
"""

import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental.layout import Layout, with_layout_constraint
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
_PASS_TOKEN = "MOE_ROE_ROW_MAJOR_REDUCE_SCATTER_CHECKS_PASSED"
_REL_TOL = 1e-5
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
# The unpatched helper: the controls below replace it.
_FORWARD_ROW_MAJOR = moe.forward_row_major


@pytest.mark.cpu_only
def test_row_major_reduce_scatter_matches_default_on_cpu_mesh():
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


def _cfg(row_major, **overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh, with the option on or off."""
  kwargs = {
      "run_name": f"moe_roe_row_major_reduce_scatter_{row_major}",
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
      # Tokamax gmm v2 is TPU-only; both runs share whichever GMM runs, so CPU uses megablox.
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
      "use_random_routing": True,
      "ring_of_experts_row_major_reduce_scatter": row_major,
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


def _layouts(jaxpr):
  """(requested major_to_minor, operand rank) of every layout_constraint equation, including nested jaxprs."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "layout_constraint":
      found.append((tuple(eqn.params["layout"].major_to_minor), eqn.invars[0].aval.ndim))
    for sub in _subjaxprs(eqn):
      found += _layouts(sub)
  return found


def _run(cfg, mesh, params, x):
  """Loss, output and gradients for `cfg`, and the layouts requested in one traced forward."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)
    loss_fn = _loss_fn(graphdef, rest)
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    layouts = _layouts(jax.make_jaxpr(loss_fn)(params, x).jaxpr)
    (loss, out), grads = jax.jit(grad_fn)(params, x)
  return jax.device_get((loss, out, grads)), layouts


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


def _case(label, overrides, helper=None):
  """Returns (comparison passed, structure ok, bit-identical). `helper` replaces `moe.forward_row_major`."""
  cfg_old, cfg_new = _cfg(False, **overrides), _cfg(True, **overrides)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  batch = int(cfg_old.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, layouts_old = _run(cfg_old, mesh, params, x)
  with mock.patch.object(moe, "forward_row_major", helper or _FORWARD_ROW_MAJOR):
    new, layouts_new = _run(cfg_new, mesh, params, x)
  chunks = cfg_new.num_moe_token_chunks
  not_row_major = [m for m, ndim in layouts_new if m != tuple(range(ndim))]
  structure = not layouts_old and len(layouts_new) == chunks and not not_row_major
  print(
      f"{label}: [{'ok' if structure else 'FAIL'}] layout constraints off/on={len(layouts_old)}/{len(layouts_new)}"
      f" (want 0/{chunks}, all row-major); requested={[m for m, _ in layouts_new]}",
      flush=True,
  )
  passed, identical = _compare(label, old, new)
  return passed, structure, identical


def _no_op(x):
  """Deliberately wrong: the option does nothing."""
  return x


def _column_major(x):
  """Deliberately wrong: requests the reversed layout, which would bring the copies back on TPU."""
  return moe._forward_layout_constraint(x, tuple(reversed(range(x.ndim))))  # pylint: disable=protected-access


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  assert "--xla_allow_excess_precision=false" in os.environ.get("XLA_FLAGS", ""), f"run with XLA_FLAGS='{_XLA_FLAGS}'"
  cases = {
      "row_major_rs_alone": _case("row_major_rs_alone", {}),
      "row_major_rs_1_chunk": _case("row_major_rs_1_chunk", {"num_moe_token_chunks": 1}),
  }
  # Each control must fail the structure check while its numbers stay bit-identical.
  controls = {
      "control_no_op": _case("control_no_op", {}, helper=_no_op),
      "control_column_major": _case("control_column_major", {}, helper=_column_major),
  }
  summary = {k: {"passed": v[0], "structure": v[1], "bit_identical": v[2]} for k, v in {**cases, **controls}.items()}
  print("SUMMARY", summary, flush=True)
  failed = [k for k, v in cases.items() if not (v[0] and v[1])]
  undetected = [k for k, v in controls.items() if v[1] or not v[2]]
  if failed or undetected:
    print(f"cases that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()


class ForwardRowMajorTest(absltest.TestCase):
  """`moe.forward_row_major` returns its input unchanged and constrains the forward value only."""

  def test_value_and_cotangent(self):
    x = jnp.arange(24.0).reshape(2, 3, 4)
    np.testing.assert_array_equal(moe.forward_row_major(x), x)
    np.testing.assert_array_equal(jax.grad(lambda a: jnp.sum(moe.forward_row_major(a) * 3.0))(x), jnp.full_like(x, 3.0))

  def test_constrains_the_forward_only(self):
    x = jnp.ones((2, 3, 4))

    def count(f):
      return len(_layouts(jax.make_jaxpr(jax.grad(lambda a: jnp.sum(f(a) * 3.0)))(x).jaxpr))

    self.assertEqual(count(moe.forward_row_major), 1)
    # A plain layout constraint also constrains its cotangent, so the check can fail.
    self.assertEqual(count(lambda a: with_layout_constraint(a, Layout(major_to_minor=(0, 1, 2)))), 2)

  def test_requires_ring_of_experts(self):
    with self.assertRaisesRegex(ValueError, "ring_of_experts_row_major_reduce_scatter=True requires use_ring_of_experts"):
      pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="moe_roe_row_major_reduce_scatter_config",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          ring_of_experts_row_major_reduce_scatter=True,
      )

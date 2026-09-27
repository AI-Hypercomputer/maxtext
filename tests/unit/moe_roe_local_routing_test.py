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
"""Ring-of-experts local routing against the all-gather-then-route path, on a CPU mesh.

With `ring_of_experts_local_routing=True`, each expert shard routes its own tokens and
all-gathers only the top-k ids and weights. Routing is per token, so this must reproduce
the default path, where every shard all-gathers the router logits and routes the whole
gathered batch.

Runs as a subprocess so the forced device count takes effect before JAX initializes.
The mesh is 8 CPU devices, fsdp1 x context4 x expert2 under cp-as-ep, so the context axis
acts as expert parallelism and the expert-parallel group is 8, as in the Qwen3.5-397B configuration
(fsdp32 x context4 x expert2). The layer is a shrunk Qwen3.5 routed MoE (16 experts, top 4)
with ragged sort, a 2x ragged buffer and two token chunks, as in that configuration. The GMM is the
megablox kernel in interpret mode, since tokamax gmm v2 does not run on CPU.

Activations are bfloat16 in every case, because the ragged sort only supports bfloat16. The
two paths differ only in how the router gradient is summed over the expert-parallel group, so:
  - The strict cases compute the router in float32 (`float32_gate_logits`, float32 input),
    which puts every difference between the paths into float32 sums. They require a
    bit-identical forward output, and the loss and every gradient (router kernel, the three
    expert kernels, the input) to agree to reduction order: max |new - old| <= 1e-5 * max |old|.
  - The bfloat16 cases keep the router in bfloat16, as in the 397B configuration. Under real routing the
    default path then sums the gradient of the gathered router logits over the group in bfloat16,
    while the local path sums the gradient of the gathered top-k weights in float32, so the router
    and input gradients differ by more than reduction order. They require a bit-identical forward
    output and each gradient's relative L2 difference below 1e-2.
Cases: random routing, real (softmax + top-k) routing, forced routing with padded and
duplicate slots, a nonzero load-balance loss, and a dropless buffer.

Two further checks make the comparison two-sided:
  - The default path must all-gather a [.., num_experts] tensor and the local path must not,
    so a switch that silently fell back to the old path cannot pass.
  - Deliberately wrong routing on the local path must FAIL the same comparison: an
    off-by-one shard offset in the random draw, expert ids shifted by one, and top-k
    results misaligned by one token.
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
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "ROE_LOCAL_ROUTING_CHECKS_PASSED"
_STRICT_REL_TOL = 1e-5
_BF16_REL_NORM_TOL = 1e-2


@pytest.mark.cpu_only
def test_roe_local_routing_matches_gathered_routing_on_cpu_mesh():
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


def _cfg(local_routing, strict, **overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh; `strict` puts the router in float32."""
  kwargs = {
      "run_name": f"roe_local_routing_{local_routing}",
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
      "float32_gate_logits": strict,
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
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": 1,
      "ici_context_parallelism": 4,
      "ici_expert_parallelism": 2,
      "context_parallel_load_balance": False,
      "ring_of_experts_local_routing": local_routing,
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
  def loss_fn(params, x, forced):
    model = nnx.merge(graphdef, params, rest)
    out, lb_loss, _ = model(x, forced_routed_experts=forced)
    loss = jnp.mean(out.astype(jnp.float32) ** 2)
    if lb_loss is not None:
      loss = loss + lb_loss.astype(jnp.float32)
    return loss, out

  return loss_fn


def _all_gather_shapes(jaxpr):
  """Output shapes of every all_gather in a jaxpr, including nested jaxprs."""
  shapes = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "all_gather":
      shapes.extend(tuple(v.aval.shape) for v in eqn.outvars)
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        # A closed jaxpr wraps the jaxpr; some JAX versions make them one class.
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          shapes.extend(_all_gather_shapes(sub))
  return shapes


def _run(cfg, mesh, params, x, forced):
  """Loss, output and gradients for `cfg` with shared parameters, and its all-gather shapes."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)
    loss_fn = _loss_fn(graphdef, rest)
    gathers = _all_gather_shapes(jax.make_jaxpr(loss_fn)(params, x, forced).jaxpr)
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x, forced)
  return jax.device_get((loss, out, grads)), gathers


def _compare(label, old, new, strict):
  """Prints the per-leaf report and returns whether the pre-registered comparison passed."""
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
    norm = float(np.linalg.norm(a))
    rel_norm = float(np.linalg.norm(b - a)) / norm if norm > 0 else diff
    frac = float(np.mean(a != b))
    # Every comparison is False on NaN, so a NaN fails.
    if strict or name == "loss":
      ok = diff <= _STRICT_REL_TOL * scale
    else:
      ok = rel_norm < _BF16_REL_NORM_TOL
    ok = ok and bool(np.isfinite(b).all()) and scale > 0
    passed &= ok
    lines.append(
        f"  [{'ok' if ok else 'FAIL'}] {name}: max|d|={diff:.3e} max|old|={scale:.3e}"
        f" rel_l2={rel_norm:.3e} differing={frac:.2%}"
    )
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _inputs(strict, forced_mode):
  """The layer input and, if `forced_mode`, forced expert ids for router replay."""
  batch = int(0.25 * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  # A float32 input keeps the router's input gradient in float32 (the experts still see bfloat16).
  x = x if strict else x.astype(jnp.bfloat16)
  if not forced_mode:
    return x, None
  # Distinct valid ids for most tokens; a padded (-1) slot and a duplicate id on others, which
  # both fall back to the router's own choice. Built per token, so any misrouting shows.
  keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
  ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
  pos = jnp.arange(_SEQ)[None, :, None]
  ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
  ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
  return x, ids.astype(jnp.int32)


def _case(label, overrides, strict=True, forced_mode=False, control=None):
  """Runs one case; `control` is a context manager applied to the local-routing run only."""
  cfg_old = _cfg(False, strict, **overrides)
  cfg_new = _cfg(True, strict, **overrides)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, forced = _inputs(strict, forced_mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, gathers_old = _run(cfg_old, mesh, params, x, forced)
  with control if control is not None else contextlib.nullcontext():
    new, gathers_new = _run(cfg_new, mesh, params, x, forced)
  logits_old = [s for s in gathers_old if s[-1] == _NUM_EXPERTS]
  logits_new = [s for s in gathers_new if s[-1] == _NUM_EXPERTS]
  structural = bool(logits_old) and not logits_new
  print(
      f"{label}: [{'ok' if structural else 'FAIL'}] [.., num_experts] all-gathers old={logits_old} new={logits_new}",
      flush=True,
  )
  return _compare(label, old, new, strict) and structural


def _off_by_one_shard():
  real = moe.random_routing

  def wrong(rng_key, gate_logits, k, num_token_shards=1, token_shard_index=0):
    return real(rng_key, gate_logits, k, num_token_shards, (token_shard_index + 1) % num_token_shards)

  return mock.patch.object(moe, "random_routing", wrong)


def _wrong_expert_ids():
  real = moe.RoutedMoE.get_topk

  def wrong(self, *args, **kwargs):
    weights, ids = real(self, *args, **kwargs)
    return weights, (ids + 1) % self.num_experts

  return mock.patch.object(moe.RoutedMoE, "get_topk", wrong)


def _misaligned_by_one_token():
  real = moe.RoutedMoE.get_topk

  def wrong(self, *args, **kwargs):
    weights, ids = real(self, *args, **kwargs)
    return jnp.roll(weights, 1, axis=1), jnp.roll(ids, 1, axis=1)

  return mock.patch.object(moe.RoutedMoE, "get_topk", wrong)


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  random_routing = {"use_random_routing": True}
  results = {
      "random": _case("random", random_routing),
      "real": _case("real", {}),
      "forced": _case("forced", {}, forced_mode=True),
      "load_balance_loss": _case("load_balance_loss", {"load_balance_loss_weight": 0.01}),
      "random_dropless": _case("random_dropless", {**random_routing, "ragged_buffer_factor": -1.0}),
      "random_bf16_router": _case("random_bf16_router", random_routing, strict=False),
      "real_bf16_router": _case("real_bf16_router", {}, strict=False),
  }
  controls = {
      "control_off_by_one_shard": _case("control_off_by_one_shard", random_routing, control=_off_by_one_shard()),
      "control_off_by_one_shard_bf16_router": _case(
          "control_off_by_one_shard_bf16_router", random_routing, strict=False, control=_off_by_one_shard()
      ),
      "control_wrong_expert_ids": _case("control_wrong_expert_ids", {}, control=_wrong_expert_ids()),
      "control_misaligned_token": _case("control_misaligned_token", {}, control=_misaligned_by_one_token()),
  }
  print("SUMMARY", {**results, **controls}, flush=True)
  failed = [k for k, v in results.items() if not v]
  undetected = [k for k, v in controls.items() if v]
  if failed or undetected:
    print(f"cases that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


class RoeLocalRoutingConfigTest(absltest.TestCase):

  def test_requires_ring_of_experts(self):
    with self.assertRaisesRegex(ValueError, "ring_of_experts_local_routing=True requires use_ring_of_experts=True"):
      pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="roe_local_routing_without_roe",
          enable_checkpointing=False,
          skip_jax_distributed_system=True,
          ring_of_experts_local_routing=True,
      )


if __name__ == "__main__":
  main()

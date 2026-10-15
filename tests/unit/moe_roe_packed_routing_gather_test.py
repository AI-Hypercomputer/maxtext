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
"""moe_roe_packed_routing_gather against the per-chunk routing gathers of the pipelined chunk loop, on a CPU mesh.

With ring_of_experts_local_routing and moe_chunk_pipeline, every token chunk routes its own tokens and all-gathers
its top-k weights and expert ids: two all-gathers per chunk. The key routes all of the layer's local tokens once and
all-gathers weights and ids as ONE float32 [batch, seq, 2 * top_k] array, which each chunk slices. Routing is per
token, so each slice holds what the chunk would have computed.

In-process checks: `pack_routing_for_all_gather` / `unpack_routing` round-trip routing weights and expert ids bit
for bit, ids up to 4095 and the -1 padding id included (the layer below has 16 experts, and bfloat16 holds every
integer up to 256, so only ids beyond 256 can tell an exact float32 packing from a lossy one), and the packed array
is float32. The validator rejects the key for each requirement it does not meet, naming it.

Runs as a subprocess so the forced device count takes effect before JAX initializes: the shrunk Qwen3.5 routed MoE
of `moe_roe_local_routing_test.py` (16 experts, top 4, 8 CPU devices under cp-as-ep, ragged sort, local routing,
two token chunks, megablox in interpret mode), with the pipelined loop in both runs. Not reachable on CPU: the
all-gather pinned to a SparseCore (moe_pin_sparse_core_all_gathers / moe_pin_sparse_core_ep_all_gathers), a TPU-only
compute_on placement; the packed gather shares it with the per-chunk gathers, and the Qwen3.5-397B RL trainer
benchmark does not pin (it measured +0.363 s per micro-batch slower with the pin).

Pre-registered comparisons, per case:
  - Bit-identical output, loss and every gradient with float32 routing weights (float32_weight_sum, the default) and
    no load-balance loss: real routing, padding excluded from routing, forced ids with a buffer that drops tokens,
    saved routing maps, with moe_routing_index_math="count", and with the benchmark's MoE settings (expert-weight
    prefetch, row-major reduce-scatter, routing maps on device, ragged_buffer_factor 2.75, prefused weights with
    fp8 fake-quant, padding excluded).
  - With the load-balance loss on, the loss and the output stay bit-identical (the per-chunk loss is computed from
    this shard's own ids of the chunk, as without the key) and the gradients agree to float32 reduction order,
    max |new - old| <= 1e-5 * max |old|: the router gradient sums the top-k and the loss terms in another order.
  - bfloat16 routing weights (float32_weight_sum=False): the output is bit-identical; their cotangent is
    reduce-scattered in float32 instead of bfloat16, so the gradients agree to relative L2 < 1e-2.
Two-sided:
  - Structure: the plain loop must all-gather two [..., top_k] arrays per chunk and the packed one none of those
    but exactly one [..., 2 * top_k] float32 array; under moe_routing_maps=device that array is tagged
    moe_routing_maps, under remat it is untagged, and nothing is ever tagged moe_route.
  - Deliberately wrong versions must FAIL: a packed gather whose result is shifted along the sequence (each chunk
    gets another chunk's routing), and the packed path with its load-balance loss dropped (what an earlier version
    of this change did).
"""

import contextlib
import os
import re
import subprocess
import sys
from unittest import mock

from absl.testing import parameterized
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
_PASS_TOKEN = "MOE_ROE_PACKED_ROUTING_GATHER_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_REL_TOL = 1e-5
_BF16_REL_NORM_TOL = 1e-2


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh, local routing, pipelined chunk loop."""
  kwargs = {
      "run_name": "moe_roe_packed_routing_gather",
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
      "moe_chunk_pipeline": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class PackRoutingTest(parameterized.TestCase):
  """The packed routing array carries weights and expert ids exactly, ids beyond bfloat16's integers included."""

  @parameterized.named_parameters(("float32_weights", jnp.float32), ("bfloat16_weights", jnp.bfloat16))
  def test_round_trip_is_exact(self, weights_dtype):
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    weights = jax.random.uniform(k0, (2, 64, 10), jnp.float32).astype(weights_dtype)
    # Every id from -1 (padding) to 4095, so the odd ids above 256 that bfloat16 cannot hold are all there.
    ids = (jax.random.permutation(k1, 2 * 64 * 10) % 4097 - 1).reshape(2, 64, 10).astype(jnp.int32)
    packed = moe.pack_routing_for_all_gather(weights, ids)
    self.assertEqual(packed.dtype, jnp.float32)
    self.assertEqual(packed.shape, (2, 64, 20))
    got_weights, got_ids = moe.unpack_routing(packed, weights.dtype, ids.dtype)
    self.assertEqual((got_weights.dtype, got_ids.dtype), (weights.dtype, ids.dtype))
    np.testing.assert_array_equal(np.asarray(got_ids), np.asarray(ids))
    np.testing.assert_array_equal(np.asarray(got_weights).view(np.uint8), np.asarray(weights).view(np.uint8))


class PackedRoutingGatherConfigTest(parameterized.TestCase):
  """The key is rejected for each requirement it does not meet; the load-balance loss is allowed."""

  @parameterized.named_parameters(
      ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
      ("no_local_routing", {"ring_of_experts_local_routing": False}, "ring_of_experts_local_routing=True"),
      ("no_pipeline", {"moe_chunk_pipeline": False}, "moe_chunk_pipeline=True"),
      ("random_routing", {"use_random_routing": True}, "use_random_routing=False"),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_roe_packed_routing_gather=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_roe_packed_routing_gather=True, **overrides)

  def test_accepted_with_load_balance_loss(self):
    cfg = _cfg(moe_roe_packed_routing_gather=True, load_balance_loss_weight=0.01)
    self.assertTrue(cfg.moe_roe_packed_routing_gather)


@pytest.mark.cpu_only
def test_packed_routing_gather_matches_per_chunk_gathers_on_cpu_mesh():
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


def _inputs(cfg, mode):
  """The layer input, forced expert ids (padded and duplicate slots) and a token mask, per `mode`."""
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  forced = token_mask = None
  if "forced" in mode:
    keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
    ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
    pos = jnp.arange(_SEQ)[None, :, None]
    ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
    ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
    forced = ids.astype(jnp.int32)
  if "padmask" in mode:
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, forced, token_mask


def _eqns(jaxpr):
  """Every equation of a jaxpr, including those of nested jaxprs."""
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          yield from _eqns(sub)


def _structure(jaxpr):
  """Routing all-gathers by last dim, and the checkpoint names on [..., 2 * top_k] arrays and anywhere."""
  eqns = list(_eqns(jaxpr))
  gathers = [tuple(v.aval.shape) for e in eqns if e.primitive.name == "all_gather" for v in e.outvars]
  names = [(e.params.get("name"), tuple(e.invars[0].aval.shape)) for e in eqns if e.primitive.name == "name"]
  return {
      "topk_gathers": sum(s[-1:] == (_TOP_K,) for s in gathers),
      "packed_gathers": sum(s[-1:] == (2 * _TOP_K,) for s in gathers),
      "packed_tags": sorted({n for n, s in names if s[-1:] == (2 * _TOP_K,)}),
      "moe_route_tags": sum(n == "moe_route" for n, _ in names),
  }


def _run(cfg, mesh, params, x, forced, token_mask):
  """Loss, output, gradients, and the structure of the traced forward."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x, forced_routed_experts=forced, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    structure = _structure(jax.make_jaxpr(loss_fn)(params, x).jaxpr)
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), structure


def _compare(label, old, new, mode):
  """Prints the per-leaf report. mode: "bitwise", "reduction_order" (gradients) or "bf16" (gradients)."""
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  named = [("output", out_old, out_new), ("loss", loss_old, loss_new)] + [
      (jax.tree_util.keystr(path), a, b)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_old)[0], jax.tree_util.tree_leaves(grads_new))
  ]
  passed, lines = True, []
  for name, a, b in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    diff, scale = float(np.max(np.abs(b - a))), float(np.max(np.abs(a)))
    rel_l2 = float(np.linalg.norm(b - a)) / max(float(np.linalg.norm(a)), 1e-30)
    same = bool(np.array_equal(a, b))
    if mode == "bitwise" or name in ("output", "loss"):
      ok = same
    elif mode == "reduction_order":
      ok = diff <= _REL_TOL * scale
    else:
      ok = rel_l2 < _BF16_REL_NORM_TOL
    ok = ok and bool(np.isfinite(b).all()) and scale > 0
    passed &= ok
    lines.append(
        f"  [{'ok' if ok else 'FAIL'}] {name}: bit-identical={same} max|d|={diff:.3e} max|old|={scale:.3e}"
        f" rel_l2={rel_l2:.3e}"
    )
  print(f"{label}: {'PASS' if passed else 'FAIL'} ({mode})", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _case(label, overrides, mode="real", compare="bitwise", control=None):
  """The per-chunk routing gathers against the packed one, on the same parameters; `control` wraps the packed run."""
  cfg_old = _cfg(**overrides)
  cfg_new = _cfg(**overrides, moe_roe_packed_routing_gather=True)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, forced, token_mask = _inputs(cfg_old, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, s_old = _run(cfg_old, mesh, params, x, forced, token_mask)
  with control if control is not None else contextlib.nullcontext():
    new, s_new = _run(cfg_new, mesh, params, x, forced, token_mask)
  n_chunks = cfg_new.num_moe_token_chunks
  want_tags = ["moe_route"] if cfg_new.moe_route != "remat" else []
  structure = (
      s_old["topk_gathers"] == 2 * n_chunks
      and s_old["packed_gathers"] == 0
      and s_new["topk_gathers"] == 0
      and s_new["packed_gathers"] == 1
      and s_new["packed_tags"] == want_tags
  )
  print(f"{label}: [{'ok' if structure else 'FAIL'}] structure plain={s_old} packed={s_new}", flush=True)
  return _compare(label, old, new, compare) and structure


def _shifted_packed_gather():
  """Deliberately wrong: the packed routing array comes back rolled along the sequence by one chunk."""
  real = jax.lax.all_gather

  def wrong(x, axis_name, **kwargs):
    out = real(x, axis_name, **kwargs)
    if out.ndim == 3 and out.shape[-1] == 2 * _TOP_K:
      # Half the LOCAL sequence: inside the shard_map the sequence is sharded over the context axis.
      out = jnp.roll(out, out.shape[1] // 2, axis=1)
    return out

  return mock.patch.object(jax.lax, "all_gather", wrong)


def _dropped_load_balance_loss():
  """Deliberately wrong: the packed path without its load-balance loss (pmean of the loss returns zeros)."""
  return mock.patch.object(jax.lax, "pmean", lambda x, axis_name, **kwargs: jnp.zeros_like(x))


# The MoE settings of the Qwen3.5-397B RL trainer benchmark that the cases above do not all set.
_BENCH_FLAGS = {
    "moe_expert_weight_prefetch": "forward",
    "ring_of_experts_row_major_reduce_scatter": True,
    "moe_route": "device",
    "ragged_buffer_factor": 2.75,
    "prefuse_moe_weights": True,
    "fp8_moe_fake_quant": True,
    "moe_exclude_padding_from_routing": True,
}


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  padded = {"moe_exclude_padding_from_routing": True}
  lb = {"load_balance_loss_weight": 0.01}
  results = {
      "real": _case("real", {}),
      "padding_excluded": _case("padding_excluded", padded, mode="padmask"),
      "forced_dropping": _case("forced_dropping", {"ragged_buffer_factor": 1.0}, mode="forced"),
      "saved_routing_maps": _case("saved_routing_maps", {"moe_route": "device"}),
      "load_balance_loss": _case("load_balance_loss", lb, compare="reduction_order"),
      "bf16_routing_weights": _case("bf16_routing_weights", {"float32_weight_sum": False}, compare="bf16"),
      "bench_flags": _case("bench_flags", _BENCH_FLAGS, mode="padmask"),
  }
  controls = {
      "control_shifted_packed_gather": _case("control_shifted_packed_gather", {}, control=_shifted_packed_gather()),
      "control_dropped_load_balance_loss": _case(
          "control_dropped_load_balance_loss", lb, compare="reduction_order", control=_dropped_load_balance_loss()
      ),
  }
  print("SUMMARY", {**results, **controls}, flush=True)
  failed = [k for k, v in results.items() if not v]
  undetected = [k for k, v in controls.items() if v]
  if failed or undetected:
    print(f"cases that failed: {failed}; positive controls that were NOT detected: {undetected}", flush=True)
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()

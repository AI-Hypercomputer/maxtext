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
"""moe_routing_index_math="count" against the default "sort", on CPU.

"count" computes the ring-of-experts ragged sort's routing index maps without sorts where it can: the expert
permutation by one packed single-operand sort (`packed_stable_sort`), the inverse permutation by counting on this
shard's slots only (`ragged_sort._local_slot_positions`, -1 elsewhere), the router-weight gradient back to slot order
by a masked gather, the gather-reduce's validity sort packed into one uint32 operand (`_preprocess(packed=True)`), and
the router's top-k indices of non-forced tokens by k argmax passes (`moe.topk_indices_by_argmax`). The permutation is
the same, so everything must be bit-identical to "sort".

What a CPU run can and cannot reach:
  - The gather-reduce's validity sort runs only on the SparseCore path. Off a TPU target `ragged_gather_reduce` takes
    its jnp reference fallback, which does not sort, so `packed_preprocess` changes nothing in the layer runs below.
    `PackedPreprocessTest` therefore calls `_preprocess` directly, the pure-JAX part of the SparseCore path.
  - Likewise the ragged gather is the jnp fallback here. Everything else ("count"'s permutation, its counted inverse,
    the masked weight-gradient gather, the argmax top-k) is plain JAX and runs in the layer below exactly as on TPU.

In-process checks:
  - `_local_slot_positions` equals argsort(argsort(ids)) - (this shard's window start) on this shard's slots and -1
    elsewhere, for every shard, with token counts that are not a multiple of the block, the padding id, and tokens
    that repeat an expert.
  - `topk_indices_by_argmax` equals `jax.lax.top_k`'s indices, ties included.
  - `_preprocess(packed=True)` equals `_preprocess(packed=False)`, every output.
  - The config validator rejects "count" off the ring-of-experts ragged sort, one case per requirement.

The layer check runs as a subprocess, so the forced device count takes effect before JAX initializes: the shrunk
Qwen3.5 routed MoE of `moe_roe_local_routing_test.py` (16 experts, top 4, 8 CPU devices under cp-as-ep, ragged sort,
local routing, two token chunks, megablox in interpret mode). Pre-registered: the forward output, the loss and every
gradient bit-identical to "sort", in each case: the plain and the pipelined chunk loop, padding excluded from routing
(every token takes the forced-routing branch, so the argmax top-k runs), forced ids with a buffer that drops tokens
(asserted to overflow), a dropless buffer, saved routing maps, and a bfloat16 router. Two-sided:
  - Structure: "count" must trace fewer `sort` equations than "sort" in the gradient and no argsort at all (a sort
    of more than one operand: jnp.argsort sorts its keys with an iota), while "sort" traces argsorts; and with
    padding excluded no `top_k`. So a switch that silently fell back, completely or for one map, cannot pass.
  - Wiring of the SparseCore-only validity sort: CPU cannot run it, but the flag that selects it reaches
    `ragged_gather_reduce` on every target. A spy records it: every call under "count" must pass
    packed_preprocess=True and every call under "sort" False.
  - The bench's MoE settings (expert-weight prefetch, row-major reduce-scatter, routing maps saved on device,
    ragged_buffer_factor 2.75, prefused weights with fp8 fake-quant, padding excluded) in one case.
  - A deliberately wrong counted inverse (every local position shifted by one) must FAIL the comparison.
  - "count" without the forward permutation must raise in the unsort backward instead of argsorting a partial inverse.
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
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.ragged import ragged_gather_reduce_v2
from maxtext.kernels.ragged import ragged_sort
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "MOE_ROUTING_INDEX_MATH_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"


def _expected_local_positions(ids, num_experts, shard, ep_size):
  """Reference: each slot's rank in the stable sort by expert id, relative to this shard's window; -1 off-shard."""
  flat = np.asarray(ids).reshape(-1)
  rank = np.argsort(np.argsort(flat, kind="stable"), kind="stable")
  local_num_experts = num_experts // ep_size
  start = shard * local_num_experts
  local = (flat >= start) & (flat < start + local_num_experts)
  window_start = int(np.sum(flat < start))
  return np.where(local, rank - window_start, -1).astype(np.int32)


@pytest.mark.cpu_only
class LocalSlotPositionsTest(parameterized.TestCase):
  """`_local_slot_positions` is the inverse permutation restricted to this shard's window."""

  @parameterized.named_parameters(
      # (num_tokens, topk, num_experts, ep_size, block): one exact block, several blocks with a partial last one,
      # fewer tokens than a block, and the production shape's ratios at a small scale.
      ("one_block", 128, 4, 16, 8, 128),
      ("partial_blocks", 300, 4, 16, 8, 64),
      ("short", 5, 2, 8, 2, 128),
      ("wide_topk", 200, 10, 64, 4, 32),
  )
  def test_matches_double_argsort(self, num_tokens, topk, num_experts, ep_size, block):
    rng = np.random.default_rng(num_tokens)
    # Ids in [0, num_experts]: num_experts is the padding id (never local). Distinct per token, as top-k gives.
    ids = np.stack([rng.permutation(num_experts + 1)[:topk] for _ in range(num_tokens)]).astype(np.int32)
    for shard in range(ep_size):
      got = np.asarray(ragged_sort._local_slot_positions(jnp.asarray(ids), num_experts, shard, ep_size, block))  # pylint: disable=protected-access
      np.testing.assert_array_equal(got, _expected_local_positions(ids, num_experts, shard, ep_size), err_msg=f"{shard=}")

  def test_repeated_expert_within_a_token(self):
    """A token that picks one expert twice: the stable order puts its earlier slot first."""
    ids = np.array([[3, 3, 1], [1, 3, 3], [16, 16, 0], [0, 1, 3]], np.int32)
    for shard in range(8):
      got = np.asarray(ragged_sort._local_slot_positions(jnp.asarray(ids), 16, shard, 8, 2))  # pylint: disable=protected-access
      np.testing.assert_array_equal(got, _expected_local_positions(ids, 16, shard, 8), err_msg=f"{shard=}")

  def test_wrong_reference_is_detected(self):
    """Two-sided: the comparison above fails on a deliberately wrong expectation (an unstable rank)."""
    ids = np.array([[1, 0], [1, 0], [0, 1]], np.int32)
    got = np.asarray(ragged_sort._local_slot_positions(jnp.asarray(ids), 2, 0, 1, 2))  # pylint: disable=protected-access
    flat = ids.reshape(-1)
    unstable_rank = np.argsort(np.argsort(-np.arange(flat.size) + flat * 100))  # ties broken by DEcreasing index
    self.assertFalse(np.array_equal(got, unstable_rank.astype(np.int32)))


@pytest.mark.cpu_only
class TopkIndicesByArgmaxTest(parameterized.TestCase):
  """`topk_indices_by_argmax` returns `jax.lax.top_k`'s indices."""

  @parameterized.named_parameters(("distinct", False), ("ties", True))
  def test_matches_top_k(self, ties):
    scores = jax.random.uniform(jax.random.PRNGKey(0), (4, 33, 64), jnp.float32)
    if ties:
      # Few distinct values, so most rows tie at the top-k boundary; and exact zeros, like underflowed softmax.
      scores = jnp.round(scores * 4) / 4
    want = jax.lax.top_k(scores, 10)[1]
    got = moe.topk_indices_by_argmax(scores, 10)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_on_softmax_probabilities(self):
    logits = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 512), jnp.float32) * 30.0  # many probs underflow to 0
    probs = jax.nn.softmax(logits, axis=-1)
    np.testing.assert_array_equal(
        np.asarray(moe.topk_indices_by_argmax(probs, 10)), np.asarray(jax.lax.top_k(probs, 10)[1])
    )


@pytest.mark.cpu_only
class PackedPreprocessTest(parameterized.TestCase):
  """The SparseCore gather-reduce's validity sort, packed and unpacked.

  Only the SparseCore path runs `_preprocess`; on a CPU target `ragged_gather_reduce` takes its reference fallback,
  so this pure-JAX function is called directly.
  """

  @parameterized.named_parameters(
      # (rows, reduce_group_size, num_row_partitions, num_simd_lanes, row_chunk_size)
      ("one_partition", 96, 4, 1, 16, 16),
      ("v7x_like", 1024, 8, 16, 16, 48),
      ("padded_chunks", 640, 10, 8, 8, 24),
  )
  def test_packed_matches_argsort(self, rows, group, partitions, lanes, chunk):
    for density in (0.0, 0.3, 0.9, 1.0):
      mask = jax.random.uniform(jax.random.PRNGKey(rows), (rows,)) < density
      want = ragged_gather_reduce_v2._preprocess(mask, group, partitions, lanes, chunk)  # pylint: disable=protected-access
      got = ragged_gather_reduce_v2._preprocess(mask, group, partitions, lanes, chunk, packed=True)  # pylint: disable=protected-access
      for name, a, b in zip(("sorted_by_validity", "rows_per_partition", "group_mask"), want, got):
        self.assertEqual(a.dtype, b.dtype, name)
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a), err_msg=f"{name} {density=}")


class RoutingIndexMathConfigTest(parameterized.TestCase):
  """moe_routing_index_math="count" is rejected where no ring-of-experts ragged sort computes routing index maps."""

  @parameterized.named_parameters(
      # One case per requirement of the validator; each must be named in the error.
      ("no_ragged_sort", {"use_ragged_sort": False}, "use_ragged_sort=True"),
      (
          "no_ring_of_experts",
          {"use_ring_of_experts": False, "num_moe_token_chunks": 1, "ring_of_experts_local_routing": False},
          "use_ring_of_experts=True",
      ),
      ("dense_matmul", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
  )
  def test_rejected_off_the_ring_ragged_sort(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_routing_index_math='count' requires: .*{re.escape(requirement)}"):
      _cfg(moe_routing_index_math="count", **overrides)

  def test_accepted_on_the_ring_ragged_sort(self):
    self.assertEqual(_cfg(moe_routing_index_math="count").moe_routing_index_math, "count")

  def test_names_every_unmet_requirement(self):
    """The error names all unmet requirements, not only the first."""
    with self.assertRaises(ValueError) as raised:
      _cfg(moe_routing_index_math="count", te_moe_block=True, use_ragged_sort=False)
    self.assertIn("te_moe_block=False", str(raised.exception))
    self.assertIn("use_ragged_sort=True", str(raised.exception))


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh (moe_roe_local_routing_test.py's, local routing)."""
  kwargs = {
      "run_name": "moe_routing_index_math",
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
      # Tokamax gmm v2 is TPU-only; both settings share whichever GMM runs, so CPU uses megablox.
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
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


@pytest.mark.cpu_only
def test_count_matches_sort_on_cpu_mesh():
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
  x = x if cfg.float32_gate_logits else x.astype(jnp.bfloat16)
  forced = token_mask = None
  if "forced" in mode:
    keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
    ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
    pos = jnp.arange(_SEQ)[None, :, None]
    ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
    ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
    forced = ids.astype(jnp.int32)
  if "padmask" in mode:
    # Each sequence ends in a run of padding of a different length.
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, forced, token_mask


def _count(jaxpr, name):
  """Number of `name` equations in a jaxpr, including nested jaxprs."""
  total = 0
  for eqn in jaxpr.eqns:
    total += eqn.primitive.name == name
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          total += _count(sub, name)
  return total


def _count_argsorts(jaxpr):
  """Number of `sort` equations with more than one operand: an argsort sorts its keys together with an iota.

  "count" sorts one packed key; the forced-routing duplicate check's `jnp.sort` is single-operand too.
  """
  total = 0
  for eqn in jaxpr.eqns:
    total += eqn.primitive.name == "sort" and len(eqn.invars) > 1
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          total += _count_argsorts(sub)
  return total


def _run(cfg, mesh, params, x, forced, token_mask):
  """Runs the layer's value_and_grad; returns (loss, output, gradients), the overflow flag and the counts.

  The counts are the gradient's sort / top_k equations and the packed_preprocess flag of every
  ragged_gather_reduce call traced.
  """
  preprocess_flags = []
  real_gather_reduce = ragged_sort.ragged_gather_reduce

  def spy(*args, **kwargs):
    preprocess_flags.append(kwargs.get("packed_preprocess", False))
    return real_gather_reduce(*args, **kwargs)

  with (
      mock.patch.object(ragged_sort, "ragged_gather_reduce", spy),
      jax.set_mesh(mesh),
      nn_partitioning.axis_rules(cfg.logical_axis_rules),
  ):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      layer = nnx.merge(graphdef, params, rest)
      out, lb_loss, _ = layer(x, forced_routed_experts=forced, token_mask=token_mask)
      sown = nnx.state(layer, nnx.Intermediate).to_pure_dict()
      flags = maxtext_utils.collect_intermediates_by_suffix(sown, "moe_has_overflow")
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, (out, jnp.any(jnp.concatenate([jnp.ravel(f) for f in flags])))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    jaxpr = jax.make_jaxpr(grad_fn)(params, x).jaxpr
    (loss, (out, overflow)), grads = jax.jit(grad_fn)(params, x)
  counts = {
      "sort": _count(jaxpr, "sort"),
      "argsort": _count_argsorts(jaxpr),
      "top_k": _count(jaxpr, "top_k"),
      "packed_preprocess": sorted(set(preprocess_flags)),
  }
  return jax.device_get((loss, out, grads)), bool(overflow), counts


def _bit_identical(label, old, new):
  """Prints the per-leaf report; True iff the output, the loss and every gradient are bit-identical."""
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  named = [("output", out_old, out_new), ("loss", loss_old, loss_new)] + [
      (jax.tree_util.keystr(path), a, b)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_old)[0], jax.tree_util.tree_leaves(grads_new))
  ]
  passed, lines = True, []
  for name, a, b in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    same = bool(np.array_equal(a, b)) and bool(np.isfinite(b).all()) and float(np.max(np.abs(a))) > 0
    passed &= same
    lines.append(f"  [{'ok' if same else 'FAIL'}] {name}: bit-identical={same} max|d|={np.max(np.abs(a - b)):.3e}")
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _case(label, overrides, mode="real", control=None, expect_overflow=False):
  """Runs "sort" and "count" on the same parameters; `control` is applied to the "count" run only."""
  cfg_sort = _cfg(**overrides)
  cfg_count = _cfg(**overrides, moe_routing_index_math="count")
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_sort), cfg_sort.mesh_axes)
  x, forced, token_mask = _inputs(cfg_sort, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_sort.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_sort, mesh), nnx.Param, ...)
  old, overflow_old, counts_old = _run(cfg_sort, mesh, params, x, forced, token_mask)
  with control if control is not None else contextlib.nullcontext():
    new, overflow_new, counts_new = _run(cfg_count, mesh, params, x, forced, token_mask)
  structure = counts_new["sort"] < counts_old["sort"]
  # "count" sorts only the packed key (one operand); "sort" argsorts with an iota operand.
  structure &= counts_new["argsort"] == 0 < counts_old["argsort"]
  # The SparseCore validity-sort flag, on every gather-reduce call (CPU runs the reference fallback, which ignores it).
  structure &= counts_new["packed_preprocess"] == [True] and counts_old["packed_preprocess"] == [False]
  if token_mask is not None:
    structure &= counts_new["top_k"] == 0 < counts_old["top_k"]
  overflow_ok = overflow_old == overflow_new == expect_overflow
  print(
      f"{label}: [{'ok' if structure else 'FAIL'}] sort/top_k equations sort={counts_old} count={counts_new};"
      f" [{'ok' if overflow_ok else 'FAIL'}] overflow sort={overflow_old} count={overflow_new}"
      f" (want {expect_overflow})",
      flush=True,
  )
  return _bit_identical(label, old, new) and structure and overflow_ok


def _shifted_local_positions():
  """Deliberately wrong: every local slot's counted position is one too far."""
  real = ragged_sort._local_slot_positions  # pylint: disable=protected-access

  def wrong(*args, **kwargs):
    pos = real(*args, **kwargs)
    return jnp.where(pos >= 0, pos + 1, pos)

  return mock.patch.object(ragged_sort, "_local_slot_positions", wrong)


def _count_needs_forward_permutation():
  """The unsort backward of "count" raises without the forward permutation (its inverse is partial)."""
  mesh = Mesh(np.array(jax.devices()[:2]), ("expert",))
  num_tokens, topk, num_experts = 8, 2, 4
  ids = jnp.asarray(np.stack([np.random.default_rng(t).permutation(num_experts)[:topk] for t in range(num_tokens)]))

  def unsort_loss(tokens):
    def body(tokens, ids):
      _, group_sizes, revert, _ = ragged_sort.ring_ragged_sort(
          tokens, ids, num_experts, topk, "expert", 2, index_math="count"
      )
      out = ragged_sort.ring_ragged_unsort(
          tokens.repeat(topk, axis=0),
          group_sizes,
          revert,
          topk,
          num_experts // 2,
          "expert",
          topk_weights=jnp.ones((num_tokens * topk,), jnp.float32),
          index_math="count",
      )
      return jnp.sum(out.astype(jnp.float32))

    return jax.shard_map(body, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)(tokens, ids)

  try:
    jax.grad(unsort_loss)(jnp.ones((num_tokens, 128), jnp.bfloat16))
  except ValueError as e:
    raised = "needs topk_argsort_indices" in str(e)
  else:
    raised = False
  print(f"count without the forward permutation: [{'ok' if raised else 'FAIL'}] the unsort backward raises", flush=True)
  return raised


# The subset of the Qwen3.5-397B RL trainer benchmark's MoE settings that this file needs (the cases above do not
# all set them).
_BENCH_FLAGS = {
    "moe_chunk_pipeline": True,
    "moe_expert_weight_prefetch": "forward",
    "ring_of_experts_row_major_reduce_scatter": True,
    "moe_routing_maps": "device",
    "ragged_buffer_factor": 2.75,
    "prefuse_moe_weights": True,
    "fp8_moe_fake_quant": True,
    "moe_exclude_padding_from_routing": True,
}


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  pipelined = {"moe_chunk_pipeline": True}
  padded = {**pipelined, "moe_exclude_padding_from_routing": True}
  results = {
      "plain_loop": _case("plain_loop", {}),
      "pipelined": _case("pipelined", pipelined),
      "padding_excluded": _case("padding_excluded", padded, mode="padmask"),
      "forced_dropping": _case(
          "forced_dropping", {**padded, "ragged_buffer_factor": 1.0}, mode="forced_padmask", expect_overflow=True
      ),
      "dropless": _case("dropless", {"ragged_buffer_factor": -1.0}, mode="forced"),
      "saved_routing_maps": _case("saved_routing_maps", {**pipelined, "moe_routing_maps": "device"}),
      "bf16_router": _case("bf16_router", {**padded, "float32_gate_logits": False}, mode="padmask"),
      "bench_flags": _case("bench_flags", _BENCH_FLAGS, mode="padmask"),
      "guard": _count_needs_forward_permutation(),
  }
  controls = {
      "control_shifted_local_positions": _case(
          "control_shifted_local_positions", pipelined, control=_shifted_local_positions()
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

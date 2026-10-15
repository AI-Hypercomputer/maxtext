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
"""moe_router_grad_from_wo_input: the down projection and the unsort as one op, against the two apart, on CPU.

`ragged_sort.ring_ragged_unsort_down_projection` runs the down-projection GEMM and then the forward of
`ring_ragged_unsort` (the same helpers), so the output must be bit-identical. Its backward does not read the GEMM's
output: it takes the GEMM's own VJP at the input w * h with the unweighted cotangent g, giving u = g wo^T and
dwo = (w * h)^T g, then dh = w * u and the routing-weight gradient <u, h> (= <g, h wo>). Each gradient keeps one
rounding to the activation dtype, at a different point, so the gradients agree with the two ops apart to bf16
rounding, not bit for bit.

What a CPU run can and cannot reach: the ragged gather and gather-reduce take their jnp reference fallbacks off a TPU
target (the SparseCore kernels, and the validity sort of the gather-reduce, are TPU-only), and the layer's GMM is
megablox in interpret mode. The fused op's own math (the custom VJP, the weighting, the dot, the masks, the slot-order
return under both index maps and both buffering modes) is plain JAX and runs here as on TPU.

Pre-registered, all in a subprocess with 8 CPU devices:
  - The op on a 4-device expert-parallel mesh, with a jnp grouped matmul as the down projection, the full buffer and
    a packed buffer that drops tokens, moe_routing_index_math "sort" and "count": the output bit-identical to the
    two ops apart, every gradient (dx, dw, dwi, dwo) within relative L2 1e-2 of theirs. With a forward tie: the tie
    comes back unchanged, its cotangent passes through, and the traced forward ties it to the GEMM output with an
    optimization barrier. "count" without the forward permutation must raise.
  - Precision, the op alone (h an input, so only the backward rounds): against the same function written densely in
    float32 (slot by slot, plain autodiff, no ragged kernels; the ragged gather-reduce returns bf16, so the two-ops
    path cannot serve as a float32 reference) at the same bf16-representable inputs, each fused gradient (dh, dw,
    dwo) must be no further than _PRECISION_RATIO = 1.10 times as far as the two-ops gradient, on 6 seeds x {full
    buffer "sort", dropping packed buffer "count"}. The routing weights sit at the bottom of a binade
    ([0.25, 0.3)), where bf16 rounds them worst, so a backward that rounds them is visible. Measured over the 12
    runs: dh 0.996-1.005, dwo 0.992-1.003, dw 0.941-1.065 (dw, one value per slot, is the noisiest), so 1.10 sits
    above the spread. A backward that rounds the routing weight to bf16 measures 1.31-1.34 on dh and dwo on every
    seed, and one that shrinks the routing-weight gradient by 0.5% measures 3.0-3.3 on dw; both must fail.
  - The shrunk Qwen3.5 routed MoE of moe_roe_local_routing_test.py (16 experts, top 4, cp-as-ep, ragged sort, local
    routing, two token chunks): output and loss bit-identical to the key off, every gradient within relative L2 1e-2
    (measured <= 4.2e-3): plain and pipelined chunk loop, dropless, padding excluded with "count", saved routing
    maps, the prefused-weight keys, and the benchmark's MoE settings with every compatible key of this series on
    against all off (moe_routing_index_math="count", moe_unsplit_prefused_wi, moe_prefused_swiglu_vjp,
    moe_roe_packed_routing_gather; expert-weight prefetch, row-major reduce-scatter, routing maps on device,
    ragged_buffer_factor 2.75, fp8 fake-quant, padding excluded). A 0.5% error in the routing-weight gradient sits
    inside the layer's 1e-2 tolerance; the precision check above is what catches it. Structure: the compiled
    jax.grad(jax.checkpoint(loss)) has exactly num_moe_token_chunks GEMM loops fewer (the down-projection GEMM is no
    longer recomputed in the backward), the pipelined forward keeps the same number of optimization barriers (tie 2
    moves into the op), and every chunk but the last reads the next chunk's routed tokens from the op's tie output.
  - On an expert 4 x tensor 2 mesh the layer runs without the key and raises at trace time when the key reaches it
    past the validator.
  - Deliberately wrong versions must FAIL: the fused backward with the routing weight dropped from the input and
    weight gradients (at the op and in the layer).
  - The validator rejects the key for each requirement it does not meet, naming it.
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
_PASS_TOKEN = "MOE_ROUTER_GRAD_FROM_WO_INPUT_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_GRAD_REL_L2 = 1e-2
# The fused backward's error against a float32 run, over the two-ops backward's error (see the module docstring).
_PRECISION_RATIO = 1.10
_PRECISION_SEEDS = range(6)


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config on the cp-as-ep mesh (moe_roe_local_routing_test.py's, local routing)."""
  kwargs = {
      "run_name": "moe_router_grad_from_wo_input",
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
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


# A mesh with tensor parallelism: the default logical axis rules (cp-as-ep has no tensor axis), expert 4 x tensor 2.
_TP_MESH = {
    "custom_mesh_and_rule": "",
    "ici_context_parallelism": 1,
    "ici_expert_parallelism": 4,
    "ici_tensor_parallelism": 2,
    "per_device_batch_size": 0.5,
}


class RouterGradFromWoInputConfigTest(parameterized.TestCase):
  """The key is rejected for each requirement it does not meet."""

  @parameterized.named_parameters(
      ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
      (
          "no_ring",
          {"use_ring_of_experts": False, "num_moe_token_chunks": 1, "ring_of_experts_local_routing": False},
          "use_ring_of_experts=True",
      ),
      ("no_ragged_sort", {"use_ragged_sort": False}, "use_ragged_sort=True"),
      ("tensor_parallel", _TP_MESH, "no tensor parallelism"),
      ("mlp_bias", {"mlp_bias": True}, "mlp_bias=False"),
      ("quantization", {"quantization": "int8"}, "no weight quantization"),
      ("tc_ragged_sort", {"moe_tc_ragged_sort": True}, "moe_tc_ragged_sort=False"),
      ("chunk_wgrad", {"moe_accumulate_chunk_wgrad": True}, "moe_accumulate_chunk_wgrad=False"),
      ("prequant_unsort", {"moe_bwd_prequant_before_unsort": True}, "moe_bwd_prequant_before_unsort=False"),
      ("quantized_token_ag", {"moe_quantize_token_all_gather": True}, "moe_quantize_token_all_gather=False"),
      ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks=0"),
      ("minimal_remat", {"remat_policy": "minimal"}, "remat_policy=minimal saves it as mlpwo"),
      ("save_dot_remat", {"remat_policy": "save_dot_except_mlpwi"}, "remat_policy=save_dot_except_mlpwi saves it"),
      ("saved_mlpwo", {"remat_policy": "custom", "mlpwo": "device"}, "remat_policy=custom saves it as mlpwo"),
      ("saved_moe_mlpwo", {"remat_policy": "custom", "moe_mlpwo": "offload"}, "saves it as moe_mlpwo"),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_router_grad_from_wo_input=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_router_grad_from_wo_input=True, **overrides)

  def test_accepted(self):
    self.assertTrue(_cfg(moe_router_grad_from_wo_input=True, moe_chunk_pipeline=True).moe_router_grad_from_wo_input)
    # A policy that saves only other tensors (here the dense MLP input of the shared expert) is fine.
    cfg = _cfg(moe_router_grad_from_wo_input=True, remat_policy="custom", mlpwi_1="device")
    self.assertTrue(cfg.moe_router_grad_from_wo_input)


@pytest.mark.cpu_only
def test_router_grad_from_wo_input_on_cpu_mesh():
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


# --------------------------------------------------------------------------------------------------- the op alone
_OP_EP, _OP_EXPERTS, _OP_TOPK, _OP_TOKENS, _OP_D, _OP_F = 4, 8, 2, 64, 128, 64
# A packed buffer smaller than the busiest shard's share of the 128 slots (32 on average), so tokens are dropped.
_OP_DROPPING_BUFFER = 24
# The precision check's op: more slots (1024, 256 per shard on average), so each error estimate is tight.
_PRECISION_TOKENS = 512
_PRECISION_DROPPING_BUFFER = 192


def _grouped_matmul(h, w, group_sizes, expert_assignments, group_offset):
  """A down projection with its own (autodiff) VJP: row r uses expert (global group of r) - group_offset."""
  del expert_assignments
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(h.shape[0])[:, None] >= ends[None, :], axis=1) - group_offset
  out = jnp.zeros((h.shape[0], w.shape[-1]), jnp.float32)
  for e in range(w.shape[0]):
    out = out + jnp.where((local == e)[:, None], h.astype(jnp.float32) @ w[e].astype(jnp.float32), 0.0)
  return out.astype(h.dtype)


def _op_inputs(seed=0, num_tokens=_OP_TOKENS):
  """Tokens, top-k expert ids, routing weights, wi, wo, the output cotangent and per-shard down-projection inputs."""
  k = jax.random.split(jax.random.PRNGKey(seed), 7)
  x = jax.random.normal(k[0], (num_tokens, _OP_D), jnp.float32)
  ids = jnp.stack([jax.random.permutation(kk, _OP_EXPERTS)[:_OP_TOPK] for kk in jax.random.split(k[1], num_tokens)])
  w = jax.nn.softmax(jax.random.normal(k[2], (num_tokens, _OP_TOPK)), axis=-1).reshape(-1)
  wi = jax.random.normal(k[3], (_OP_EXPERTS, _OP_D, _OP_F)) * 0.1
  wo = jax.random.normal(k[4], (_OP_EXPERTS, _OP_F, _OP_D)) * 0.1
  cot = jax.random.normal(k[5], (num_tokens, _OP_D), jnp.float32)
  # For the precision check: the down-projection input of every slot, on every shard.
  h = jax.random.normal(k[6], (_OP_EP, num_tokens * _OP_TOPK, _OP_F), jnp.float32)
  return x, ids.astype(jnp.int32), w, wi, wo, cot, h


def _expert_operands(gs, buffer_rows, num_slots, local):
  """The grouped matmul's operands for this shard's buffer: (group sizes, None, group offset)."""
  shard = jax.lax.axis_index("ep")
  if buffer_rows < num_slots:
    # Packed buffer: this shard's groups from row 0, clipped to the buffer.
    ends = jnp.minimum(jnp.cumsum(jax.lax.dynamic_slice_in_dim(gs, shard * local, local)), buffer_rows)
    return (ends - jnp.concatenate([jnp.zeros((1,), ends.dtype), ends[:-1]]), None, 0)
  return (gs, None, shard * local)


def _op_run(buffer_size, index_math, fused, tie=False):
  """bf16 run on a 4-device EP mesh; returns (output, dx, dw, dwi, dwo) summed / gathered over the shards, and the
  forward-tie checks (tie unchanged, cotangent passed through)."""
  x, ids, w, wi, wo, cot, _ = _op_inputs()
  ep, local = _OP_EP, _OP_EXPERTS // _OP_EP
  mesh = Mesh(np.array(jax.devices()[:ep]), ("ep",))
  wi_s = wi.reshape(ep, local, _OP_D, _OP_F).astype(jnp.bfloat16)
  wo_s = wo.reshape(ep, local, _OP_F, _OP_D).astype(jnp.bfloat16)
  tie_value = jnp.arange(8.0, dtype=jnp.float32)

  def body(x, w, wi, wo, cot):
    wi, wo = wi[0], wo[0]

    def f(x, w, wi, wo, t):
      sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
          x, ids, _OP_EXPERTS, _OP_TOPK, "ep", ep, buffer_size=buffer_size, index_math=index_math
      )
      operands = _expert_operands(gs, sorted_x.shape[0], _OP_TOKENS * _OP_TOPK, local)
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
            _OP_TOPK,
            local,
            "ep",
            topk_weights=w,
            topk_argsort_indices=perm_arg,
            forward_tie=t if tie else None,
            index_math=index_math,
        )
      y = _grouped_matmul(h, wo, *operands)
      out = ragged_sort.ring_ragged_unsort(
          y, gs, revert, _OP_TOPK, local, "ep", topk_weights=w, topk_argsort_indices=perm_arg, index_math=index_math
      )
      return (out, t) if tie else out

    def g(x, w, wi, wo, t):
      res = f(x, w, wi, wo, t)
      return res if tie else (res, t)

    (out, t_out), vjp = jax.vjp(g, x, w, wi, wo, tie_value)
    gx, gw, gwi, gwo, gt = vjp((cot.astype(out.dtype), 2.0 * t_out))
    return out[None], gx[None], gw[None], gwi[None], gwo[None], t_out[None], gt[None]

  sm = jax.shard_map(
      body, mesh=mesh, in_specs=(P(), P(), P("ep"), P("ep"), P()), out_specs=(P("ep"),) * 7, check_vma=False
  )
  args = (x.astype(jnp.bfloat16), w, wi_s, wo_s, cot)
  outs = [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.jit(sm)(*args)]
  out, gx, gw, gwi, gwo, t_out, gt = outs
  arrays = [
      out.sum(0),
      gx.sum(0),
      gw.sum(0),
      gwi.reshape(_OP_EXPERTS, _OP_D, _OP_F),
      gwo.reshape(_OP_EXPERTS, _OP_F, _OP_D),
  ]
  tie_ok = bool(np.array_equal(t_out[0], np.asarray(tie_value)))
  tie_ok &= bool(np.array_equal(gt[0], 2.0 * np.asarray(tie_value)))
  barrier = None
  if fused and tie:
    jaxpr = jax.make_jaxpr(sm)(*args).jaxpr
    barrier = any(_barrier_ties_shapes(eqn) for eqn in _eqns(jaxpr))
  return arrays, tie_ok, barrier


def _barrier_ties_shapes(eqn):
  """An optimization barrier whose operands include the tie (shape (8,)) next to a [rows, D] GEMM output."""
  if eqn.primitive.name != "optimization_barrier":
    return False
  shapes = [tuple(v.aval.shape) for v in eqn.invars]
  return (8,) in shapes and any(len(s) == 2 and s[-1] == _OP_D for s in shapes)


def _rel(value, reference):
  """Relative L2 distance of `value` from `reference`."""
  return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def _op_case(label, buffer_size, index_math, tie=False, control=None):
  """The fused op against the two ops apart; `control` wraps the fused run."""
  base, _, _ = _op_run(buffer_size, index_math, False, tie)
  with control if control is not None else contextlib.nullcontext():
    new, tie_ok, barrier = _op_run(buffer_size, index_math, True, tie)
  passed = bool(np.array_equal(new[0], base[0]))
  lines = [f"  [{'ok' if passed else 'FAIL'}] output bit-identical={passed}"]
  for name, a, b in zip(("dx", "dw", "dwi", "dwo"), base[1:], new[1:]):
    rel = _rel(b, a)
    ok = rel < _GRAD_REL_L2 and bool(np.isfinite(b).all())
    passed &= ok
    lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: rel_l2 vs two ops={rel:.2e}")
  if tie:
    passed &= tie_ok and bool(barrier)
    lines.append(
        f"  [{'ok' if tie_ok and barrier else 'FAIL'}] forward tie unchanged, cotangent passed: {tie_ok};"
        f" barrier ties it to the GEMM output: {barrier}"
    )
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _dense_unsort_down_projection(h, w, wo, ids, gs, revert, shard, local):
  """The fused op's function written densely, slot by slot (no ragged kernels, no rounding in float32).

  Slot i (token i // top_k) reads buffer row `revert[i]` (packed buffer: minus this shard's window start, and only
  below the buffer's row count), multiplies it by its expert's down projection, and adds it, weighted, to its token.
  """
  num_slots, buffer_rows = revert.shape[0], h.shape[0]
  local_expert = ids.reshape(-1) - shard * local
  is_local = (local_expert >= 0) & (local_expert < local)
  start = jnp.sum(jnp.where(jnp.arange(gs.shape[0]) < shard * local, gs, 0))
  row = revert if buffer_rows >= num_slots else revert - start
  limit = buffer_rows if buffer_rows >= num_slots else jnp.minimum(jnp.sum(jnp.where(is_local, 1, 0)), buffer_rows)
  valid = is_local & (row >= 0) & (row < limit)
  h_slot = jnp.where(valid[:, None], h[jnp.where(valid, row, 0)], 0.0)
  y_slot = jnp.einsum("nf,nfd->nd", h_slot, wo[jnp.clip(local_expert, 0, local - 1)])
  weighted = jnp.where(valid[:, None], (w[:, None] * y_slot), 0.0)
  return jnp.sum(weighted.reshape(num_slots // _OP_TOPK, _OP_TOPK, -1), axis=1)


def _op_backward_run(seed, buffer_size, index_math, path):
  """The op (path "fused"), the two ops apart ("two_ops", both bf16) or the dense float32 function ("reference"),
  differentiated at (h, w, wo) with the cotangent g on a 4-device EP mesh. h is an input, so the only roundings are
  the backward's own (and, in the two-ops path, the forward GEMM output's). Returns (dh, dw, dwo), summed or gathered
  over the shards. The routing weights are in [0.25, 0.3), the bottom of a binade, where bf16 rounds them worst."""
  x, ids, _, _, wo, cot, h = _op_inputs(seed, _PRECISION_TOKENS)
  w = jax.random.uniform(jax.random.PRNGKey(1000 + seed), (_PRECISION_TOKENS * _OP_TOPK,), jnp.float32, 0.25, 0.3)
  ep, local = _OP_EP, _OP_EXPERTS // _OP_EP
  mesh = Mesh(np.array(jax.devices()[:ep]), ("ep",))
  # The same bf16-representable inputs on every path; the reference only computes in float32.
  dtype = jnp.float32 if path == "reference" else jnp.bfloat16
  wo_s = wo.reshape(ep, local, _OP_F, _OP_D).astype(jnp.bfloat16).astype(dtype)
  h_s = h.astype(jnp.bfloat16).astype(dtype)
  cot = cot.astype(jnp.bfloat16).astype(dtype)

  def body(h, w, wo, cot):
    h, wo = h[0], wo[0]
    sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
        x.astype(jnp.bfloat16), ids, _OP_EXPERTS, _OP_TOPK, "ep", ep, buffer_size=buffer_size, index_math=index_math
    )
    rows = sorted_x.shape[0]
    operands = _expert_operands(gs, rows, _PRECISION_TOKENS * _OP_TOPK, local)
    perm_arg = perm if index_math == "count" else None

    def f(h, w, wo):
      if path == "reference":
        return _dense_unsort_down_projection(h, w, wo, ids, gs, revert, jax.lax.axis_index("ep"), local)
      if path == "fused":
        return ragged_sort.ring_ragged_unsort_down_projection(
            h,
            wo,
            _grouped_matmul,
            operands,
            gs,
            revert,
            _OP_TOPK,
            local,
            "ep",
            topk_weights=w,
            topk_argsort_indices=perm_arg,
            index_math=index_math,
        )
      y = _grouped_matmul(h, wo, *operands)
      return ragged_sort.ring_ragged_unsort(
          y, gs, revert, _OP_TOPK, local, "ep", topk_weights=w, topk_argsort_indices=perm_arg, index_math=index_math
      )

    out, vjp = jax.vjp(f, h[:rows], w, wo)
    dh, dw, dwo = vjp(cot.astype(out.dtype))
    return dh[None], dw[None], dwo[None]

  sm = jax.shard_map(body, mesh=mesh, in_specs=(P("ep"), P(), P("ep"), P()), out_specs=(P("ep"),) * 3, check_vma=False)
  dh, dw, dwo = [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.jit(sm)(h_s, w, wo_s, cot)]
  return dh, dw.sum(0), dwo.reshape(_OP_EXPERTS, _OP_F, _OP_D)


def _op_precision_case(label, buffer_size, index_math, control=None):
  """Fused vs two-ops error against a float32 run, over _PRECISION_SEEDS; `control` wraps the fused runs."""
  passed, lines, ratios = True, [], []
  for seed in _PRECISION_SEEDS:
    reference = _op_backward_run(seed, buffer_size, index_math, "reference")
    two_ops = _op_backward_run(seed, buffer_size, index_math, "two_ops")
    with control if control is not None else contextlib.nullcontext():
      fused = _op_backward_run(seed, buffer_size, index_math, "fused")
    parts = []
    for name, r, a, b in zip(("dh", "dw", "dwo"), reference, two_ops, fused):
      err_two_ops, err_fused = _rel(a, r), _rel(b, r)
      ratio = err_fused / err_two_ops
      ok = ratio <= _PRECISION_RATIO and bool(np.isfinite(b).all()) and err_two_ops > 0
      passed &= ok
      ratios.append(ratio)
      parts.append(f"{name} {err_two_ops:.2e}/{err_fused:.2e}={ratio:.3f}{'' if ok else ' FAIL'}")
    lines.append(f"  seed {seed}: error vs float32, two ops / fused = ratio: {'; '.join(parts)}")
  print(
      f"{label}: {'PASS' if passed else 'FAIL'} (ratio <= {_PRECISION_RATIO}; over the seeds min"
      f" {min(ratios):.3f} max {max(ratios):.3f})",
      flush=True,
  )
  print("\n".join(lines), flush=True)
  return passed


def _count_needs_forward_permutation():
  try:
    _op_run_count_without_permutation()
  except ValueError as e:
    raised = "needs topk_argsort_indices" in str(e)
  else:
    raised = False
  print(f"count without the forward permutation: [{'ok' if raised else 'FAIL'}] the fused backward raises", flush=True)
  return raised


def _op_run_count_without_permutation():
  """Differentiates the fused op under "count" without passing the forward permutation (must raise)."""
  x, ids, w, _, wo, cot, _ = _op_inputs()
  mesh = Mesh(np.array(jax.devices()[:2]), ("ep",))

  def body(x, wo, cot):
    def f(x, wo):
      sorted_x, gs, revert, _ = ragged_sort.ring_ragged_sort(x, ids, _OP_EXPERTS, _OP_TOPK, "ep", 2, index_math="count")
      return ragged_sort.ring_ragged_unsort_down_projection(
          sorted_x[:, :_OP_F],
          wo[:4],
          _grouped_matmul,
          (gs, None, 0),
          gs,
          revert,
          _OP_TOPK,
          4,
          "ep",
          topk_weights=w,
          index_math="count",
      )

    out, vjp = jax.vjp(f, x, wo)
    return vjp(cot.astype(out.dtype))[1]

  jax.shard_map(body, mesh=mesh, in_specs=(P(), P(), P()), out_specs=P(), check_vma=False)(
      x.astype(jnp.bfloat16), wo.astype(jnp.bfloat16), cot
  )


def _fan_out_changed(weight=None, slot_value=None):
  """Deliberately wrong: the fused op's backward fan-out with its routing weights or its per-slot result changed.

  ring_ragged_unsort uses the same helper, so only wrap the fused run with this.
  """
  real = ragged_sort._ring_unsort_fan_out  # pylint: disable=protected-access

  def wrong(*args, **kwargs):
    gathered, weight_for_sorted, mask, to_slot_order = real(*args, **kwargs)
    if weight is not None:
      weight_for_sorted = weight(weight_for_sorted)
    if slot_value is not None:
      return gathered, weight_for_sorted, mask, lambda per_row: to_slot_order(slot_value(per_row))
    return gathered, weight_for_sorted, mask, to_slot_order

  return mock.patch.object(ragged_sort, "_ring_unsort_fan_out", wrong)


def _weight_dropped_from_backward():
  """Deliberately wrong: the fused backward's fan-out reports every routing weight as 1."""
  return _fan_out_changed(weight=jnp.ones_like)


def _weight_rounded_to_bf16():
  """Deliberately wrong: the fused backward uses the routing weight rounded to bf16."""
  return _fan_out_changed(weight=lambda w: w.astype(jnp.bfloat16).astype(jnp.float32))


def _router_grad_shrunk():
  """Deliberately wrong: the fused backward's routing-weight gradient 0.5% too small."""
  return _fan_out_changed(slot_value=lambda dot: dot * 0.995)


# --------------------------------------------------------------------------------------------------- the layer
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
  """The layer input, forced expert ids and a token mask, per `mode`."""
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  forced = token_mask = None
  if "forced" in mode:
    keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
    ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
    forced = ids.astype(jnp.int32)
  if "padmask" in mode:
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, forced, token_mask


def _eqns(jaxpr):
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          yield from _eqns(sub)


def _compiled_gemm_loops(hlo_text):
  """Number of while instructions in a compiled HLO module.

  The GMMs are megablox kernels in interpret mode, each a loop over its tiles; identical GEMMs share one body
  computation, so the loop instructions (one per call), not the dots in the bodies, count the GEMMs.
  """
  return sum(" while(" in line for line in hlo_text.splitlines())


def _sub_jaxprs(eqn):
  for param in eqn.params.values():
    for sub in param if isinstance(param, (list, tuple)) else (param,):
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      if hasattr(sub, "eqns"):
        yield sub


def _tie_reads(jaxpr):
  """(fused ops with a forward tie, how many of them have their tie output read later), over all sub-jaxprs.

  A fused op with a tie is a custom_vjp call with two outputs whose primal holds an optimization barrier and the
  unsort's ragged_gather_reduce (barrier_tie's own custom_vjp holds only the barrier). Its second output is the next
  chunk's routed tokens after the barrier: if nothing reads it, that chunk is no longer ordered after this chunk's down
  projection.
  """
  ties = read = 0
  for i, eqn in enumerate(jaxpr.eqns):
    call = eqn.params.get("call_jaxpr")
    call = getattr(call, "jaxpr", call)
    if (
        eqn.primitive.name.startswith("custom_vjp_call")
        and len(eqn.outvars) == 2
        and hasattr(call, "eqns")
        and any(e.primitive.name == "optimization_barrier" for e in _eqns(call))
        and any(e.params.get("name") == "ragged_gather_reduce" for e in _eqns(call))
    ):
      ties += 1
      tie_out = eqn.outvars[-1]
      read += any(tie_out in later.invars for later in jaxpr.eqns[i + 1 :]) or tie_out in jaxpr.outvars
    for sub in _sub_jaxprs(eqn):
      sub_ties, sub_read = _tie_reads(sub)
      ties, read = ties + sub_ties, read + sub_read
  return ties, read


def _run(cfg, mesh, params, x, forced, token_mask):
  """Loss, output, gradients; forward barriers and fused-op ties; GEMM loops of the compiled rematerialized
  backward."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x, forced_routed_experts=forced, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    forward = jax.make_jaxpr(loss_fn)(params, x).jaxpr
    barriers = sum(e.primitive.name == "optimization_barrier" for e in _eqns(forward))
    ties = _tie_reads(forward)

    def linear_loss(params, x):
      # Linear in the layer output, as behind the decoder's residual add: the layer's own backward does not need its
      # output, so only the backward's own reads of the down-projection output keep that GEMM in the replay.
      out, _, _ = nnx.merge(graphdef, params, rest)(x, forced_routed_experts=forced, token_mask=token_mask)
      return jnp.sum(out.astype(jnp.float32) * jax.random.normal(jax.random.PRNGKey(3), out.shape, jnp.float32))

    remat = jax.grad(jax.checkpoint(linear_loss, policy=jax.checkpoint_policies.nothing_saveable), argnums=(0, 1))
    loops = _compiled_gemm_loops(jax.jit(remat).lower(params, x).compile().as_text())
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), barriers, ties, loops


def _layer_case(label, overrides, mode="real", control=None, keys=None, same_barriers=True):
  """The layer with `keys` off and on (default: this key alone), on the same parameters; `control` wraps the run
  with them on. `same_barriers`: the forward keeps its optimization barriers (False where other keys change them)."""
  keys = {"moe_router_grad_from_wo_input": True} if keys is None else keys
  cfg_old = _cfg(**overrides)
  cfg_new = _cfg(**overrides, **keys)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, forced, token_mask = _inputs(cfg_old, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, barriers_old, ties_old, loops_old = _run(cfg_old, mesh, params, x, forced, token_mask)
  with control if control is not None else contextlib.nullcontext():
    new, barriers_new, ties_new, loops_new = _run(cfg_new, mesh, params, x, forced, token_mask)
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  n_chunks = cfg_new.num_moe_token_chunks
  # One GEMM loop fewer per chunk in the rematerialized backward: the down projection is not replayed.
  loops_ok = loops_old - loops_new == n_chunks
  barriers_ok = barriers_new == barriers_old or not same_barriers
  # Pipelined: every chunk but the last ties its down projection to the next chunk's routed tokens, which are read.
  want_ties = (n_chunks - 1, n_chunks - 1) if cfg_new.moe_chunk_pipeline else (0, 0)
  ties_ok = ties_old == (0, 0) and ties_new == want_ties
  passed = loops_ok and barriers_ok and ties_ok
  lines = [
      f"  [{'ok' if loops_ok else 'FAIL'}] compiled remat-backward GEMM loops off/on={loops_old}/{loops_new}"
      f" (want {n_chunks} fewer)",
      f"  [{'ok' if barriers_ok else 'FAIL'}] forward barriers off/on={barriers_old}/{barriers_new}"
      f"{'' if same_barriers else ' (not compared)'}",
      f"  [{'ok' if ties_ok else 'FAIL'}] fused-op ties (count, read) off/on={ties_old}/{ties_new} (want {want_ties})",
  ]
  for name, a, b in (("output", out_old, out_new), ("loss", loss_old, loss_new)):
    same = bool(np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32)))
    passed &= same
    lines.append(f"  [{'ok' if same else 'FAIL'}] {name}: bit-identical={same}")
  for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_old)[0], jax.tree_util.tree_leaves(grads_new)):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    rel = _rel(b, a)
    ok = rel < _GRAD_REL_L2 and bool(np.isfinite(b).all())
    passed &= ok
    lines.append(f"  [{'ok' if ok else 'FAIL'}] {jax.tree_util.keystr(path)}: rel_l2={rel:.2e}")
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


class _ConfigOverride:
  """`config` with some attributes replaced: reaches the layer's trace-time checks past the config validator."""

  def __init__(self, config, **overrides):
    self._config = config
    self._overrides = overrides

  def __getattr__(self, name):  # only called for names that are not set on the instance
    if name.startswith("__") or "_config" not in self.__dict__:
      raise AttributeError(name)
    return self._overrides[name] if name in self._overrides else getattr(self._config, name)


def _tensor_parallel_guard():
  """On a tensor-parallel mesh the layer refuses the key at trace time; without it the layer runs."""
  cfg = _cfg(**_TP_MESH)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  x, forced, token_mask = _inputs(cfg, "real")
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg, mesh), nnx.Param, ...)
  (loss, _, _), _, _, _ = _run(cfg, mesh, params, x, forced, token_mask)
  plain_runs = bool(np.isfinite(loss))
  try:
    _run(_ConfigOverride(cfg, moe_router_grad_from_wo_input=True), mesh, params, x, forced, token_mask)
  except ValueError as e:
    raised = "does not support tensor parallelism in the MoE layer" in str(e)
  else:
    raised = False
  ok = plain_runs and raised
  print(
      f"tensor_parallel_guard: [{'ok' if ok else 'FAIL'}] mesh {dict(mesh.shape)}: plain layer runs: {plain_runs};"
      f" the key past the validator raises at trace time: {raised}",
      flush=True,
  )
  return ok


# The MoE settings of the Qwen3.5-397B RL trainer benchmark that the cases above do not all set.
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
# Every key of this series that composes with the others on the megablox path the CPU runs (the fused SiLU GMM is
# tokamax-gmm_v2-only; moe_wi_gmm_fused_swiglu_test.py stacks it with the others using kernel stand-ins).
_ALL_KEYS = {
    "moe_routing_index_math": "count",
    "moe_unsplit_prefused_wi": True,
    "moe_prefused_swiglu_vjp": True,
    "moe_roe_packed_routing_gather": True,
    "moe_router_grad_from_wo_input": True,
}


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  pipelined = {"moe_chunk_pipeline": True}
  dropping, precision_dropping = _OP_DROPPING_BUFFER, _PRECISION_DROPPING_BUFFER
  results = {
      "op_full_buffer_sort": _op_case("op_full_buffer_sort", None, "sort"),
      "op_full_buffer_count": _op_case("op_full_buffer_count", None, "count"),
      "op_dropping_buffer_sort": _op_case("op_dropping_buffer_sort", dropping, "sort"),
      "op_dropping_buffer_count": _op_case("op_dropping_buffer_count", dropping, "count"),
      "op_forward_tie": _op_case("op_forward_tie", dropping, "sort", tie=True),
      "op_count_guard": _count_needs_forward_permutation(),
      "op_precision_full_buffer_sort": _op_precision_case("op_precision_full_buffer_sort", None, "sort"),
      "op_precision_dropping_buffer_count": _op_precision_case(
          "op_precision_dropping_buffer_count", precision_dropping, "count"
      ),
      "layer_plain_loop": _layer_case("layer_plain_loop", {}),
      "layer_pipelined": _layer_case("layer_pipelined", pipelined),
      "layer_dropless": _layer_case("layer_dropless", {"ragged_buffer_factor": -1.0}, mode="forced"),
      "layer_count_padding_excluded": _layer_case(
          "layer_count_padding_excluded",
          {**pipelined, "moe_routing_index_math": "count", "moe_exclude_padding_from_routing": True},
          mode="padmask",
      ),
      "layer_saved_routing_maps": _layer_case("layer_saved_routing_maps", {**pipelined, "moe_routing_maps": "device"}),
      "layer_prefused_keys": _layer_case(
          "layer_prefused_keys",
          {**pipelined, "prefuse_moe_weights": True, "moe_unsplit_prefused_wi": True, "moe_prefused_swiglu_vjp": True},
      ),
      "layer_bench_flags_all_keys": _layer_case(
          "layer_bench_flags_all_keys", _BENCH_FLAGS, mode="padmask", keys=_ALL_KEYS, same_barriers=False
      ),
      "tensor_parallel_guard": _tensor_parallel_guard(),
  }
  controls = {
      "control_op_weight_dropped": _op_case(
          "control_op_weight_dropped", dropping, "sort", control=_weight_dropped_from_backward()
      ),
      "control_op_precision_weight_rounded_to_bf16": _op_precision_case(
          "control_op_precision_weight_rounded_to_bf16", None, "sort", control=_weight_rounded_to_bf16()
      ),
      "control_op_precision_router_grad_shrunk": _op_precision_case(
          "control_op_precision_router_grad_shrunk", precision_dropping, "count", control=_router_grad_shrunk()
      ),
      "control_layer_weight_dropped": _layer_case(
          "control_layer_weight_dropped", pipelined, control=_weight_dropped_from_backward()
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

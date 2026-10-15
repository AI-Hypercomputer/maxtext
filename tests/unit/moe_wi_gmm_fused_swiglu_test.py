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
"""moe_wi_gmm_fused_swiglu on CPU: megablox ops.gmm_silu's own code, its wiring in the MoE layer, and the validator.

With the key, the up projection calls megablox `gmm_silu` (the tokamax gmm_v2 kernel with the SwiGLU in its epilogue)
instead of `gmm` followed by the activation. The kernels are Pallas TPU kernels and do not run on CPU; their own test,
megablox_gmm_silu_test.py, is TPU-only (not yet run on TPU). Two CPU checks cover everything around them:

ops.gmm_silu (in-process, `GmmSiluVjpTest`). Only the Pallas entry `gmm_v2.gmm_v2` and gmm's TPU backward
`ops._gmm_bwd` are replaced: the first by a jnp grouped matmul that honours fuse_act and return_pre_act and records the
tile it is given, the second by a stand-in that records how it is called and passes its cotangent through as the input
gradient. The rest is the real ops.py: gmm_silu, its custom VJP, the forward tile (tiling[2] halved: the fused kernel's
tile_n counts the columns of each half), the residuals, and the backward, which must hand gmm's backward [d gate |
d up] of silu(gate) * up on the bf16 halves, bit for bit, with this op's DLHS options bound to the right keyword
arguments.

The layer (a subprocess, so the forced device count applies): the shrunk Qwen3.5 routed MoE of
moe_roe_local_routing_test.py with prefuse_moe_weights on 8 CPU devices (cp-as-ep, ragged sort, local routing, two
token chunks). `gmm` is replaced by a grouped matmul over the same group sizes and group offset, and `gmm_silu` by the
unfused composition of that same stand-in, i.e. silu(gate) * up of its bf16 output. The layer with the key must then
reproduce the layer without it bit for bit (the output, the loss and every gradient): that checks what the key
changes in moe.py (the rows and the routing operands handed to the fused GMM, the padding to the forward tile and its
removal, the whole [w0 | w1] weight, and the skipped activation), not the kernel. The fused kernel's real numerics
differ by rounding (silu reads the f32 accumulator). Cases: the plain and the pipelined chunk loop, a dropless buffer
(where each shard's group offset is not 0), padding excluded from routing, the whole prefused wi
(moe_unsplit_prefused_wi), with it and the other MoE keys (moe_routing_index_math="count",
moe_roe_packed_routing_gather), and with those and the benchmark's MoE settings
(expert-weight prefetch, row-major reduce-scatter, routing maps on device, ragged_buffer_factor 2.75, fp8 fake-quant,
moe_gmm_v2_dlhs_transpose_rhs). Two-sided: the fused path must trace exactly one gmm_silu call per chunk and the plain
one none; gmm_silu must get the routing operands, the wi tiling the plain path gives gmm, and the DLHS options of the
config (moe_gmm_v2_dlhs_transpose_rhs is True in the benchmark case and False elsewhere); stand-ins that swap the gate
and up halves, or that ignore the group offset, must FAIL.

Key off, not compared with anything this change touches: the full-buffer ring-of-experts ragged sort (dropless), whose
GMM routing operands get_gmm_operands_for_local_experts now builds for both paths, must match the ring of experts
without the ragged sort (use_ragged_sort=False), and a GMM that ignores the group offset must fail that comparison.

The validator rejects the key for each requirement it does not meet, naming it.
"""

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
from maxtext.kernels import megablox
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_SEQ = 64
_PASS_TOKEN = "MOE_WI_GMM_FUSED_SWIGLU_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
# The ragged-sort and plain ring paths combine the same expert outputs, possibly in another order: bf16 rounding.
_UNSORTED_REL_L2 = 1e-2


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config with prefused weights and the tokamax gmm_v2 path the key requires."""
  kwargs = {
      "run_name": "moe_wi_gmm_fused_swiglu",
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
      "use_tokamax_gmm": True,
      "use_gmm_v2": True,
      "prefuse_moe_weights": True,
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


class FusedSwigluConfigTest(parameterized.TestCase):
  """The key is rejected for each requirement of the fused gmm_v2 epilogue it does not meet."""

  @parameterized.named_parameters(
      ("not_prefused", {"prefuse_moe_weights": False}, "prefuse_moe_weights=True"),
      ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("te_moe_block", {"te_moe_block": True}, "te_moe_block=False"),
      ("vllm_rpa", {"attention": "vllm_rpa"}, "an attention other than vllm_rpa*"),
      ("no_gmm_v2", {"use_gmm_v2": False}, "use_tokamax_gmm=True and use_gmm_v2=True"),
      ("no_tokamax", {"use_tokamax_gmm": False}, "use_tokamax_gmm=True and use_gmm_v2=True"),
      ("quantization", {"quantization": "int8"}, "no weight quantization"),
      ("gelu", {"mlp_activations": ["gelu", "linear"]}, "mlp_activations=['silu', 'linear']"),
      ("gpt_oss", {"decoder_block": "gpt_oss"}, "a decoder_block other than gpt_oss"),
      ("clipped", {"mlp_activations_limit": 7.0}, "mlp_activations_limit <= 0 (no clipping)"),
      ("mlp_bias", {"mlp_bias": True}, "mlp_bias=False"),
      ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks=0"),
      ("flat_fsdp", {"moe_flat_fsdp_weights": True}, "moe_flat_fsdp_weights=False"),
      ("tc_3d_gmm", {"moe_tc_ragged_3d_gmm": True}, "moe_tc_ragged_3d_gmm=False"),
      ("quantized_token_ag", {"moe_quantize_token_all_gather": True}, "moe_quantize_token_all_gather=False"),
      ("chunk_wgrad", {"moe_accumulate_chunk_wgrad": True}, "moe_accumulate_chunk_wgrad=False"),
      ("minimal_remat", {"remat_policy": "minimal"}, "remat_policy=minimal saves it as mlpwi_0"),
      ("saved_gate", {"remat_policy": "custom", "mlpwi_0": "device"}, "remat_policy=custom saves it as mlpwi_0"),
      ("offloaded_moe_up", {"remat_policy": "custom", "moe_mlpwi_1": "offload"}, "saves it as moe_mlpwi_1"),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_wi_gmm_fused_swiglu=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_wi_gmm_fused_swiglu=True, **overrides)

  def test_fused_activation_rejected_next_to_it(self):
    with self.assertRaisesRegex(ValueError, "moe_prefused_swiglu_vjp=True requires: .*moe_wi_gmm_fused_swiglu=False"):
      _cfg(moe_wi_gmm_fused_swiglu=True, moe_prefused_swiglu_vjp=True)

  def test_accepted(self):
    self.assertTrue(_cfg(moe_wi_gmm_fused_swiglu=True, moe_unsplit_prefused_wi=True).moe_wi_gmm_fused_swiglu)
    # remat_policy=full saves nothing, and the shared expert's mlpwi_1 is not a routed-expert tensor.
    for overrides in ({"remat_policy": "full"}, {"remat_policy": "custom", "mlpwi_1": "device"}):
      self.assertTrue(_cfg(moe_wi_gmm_fused_swiglu=True, **overrides).moe_wi_gmm_fused_swiglu)


_OP_M, _OP_K, _OP_N = 64, 256, 128  # the op case: lhs [M, K], rhs [groups, K, 2N]; K == 2N, so a pass-through dlhs fits
_OP_GROUP_SIZES = np.array([20, 0, 30, 10], np.int32)  # sum 60 < M: padding rows past the last group
_OP_TILING = (64, 256, 256, 64, 256, 256, 64, 256, 256)  # gmm's 9-tuple; the fused forward tile is (64, 256, 128)


def _grouped_f32(lhs, rhs, group_sizes, group_offset):
  """f32 lhs @ rhs[local expert of row]: rows of the groups [group_offset, group_offset + groups), others 0."""
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= ends[None, :], axis=1) - jnp.reshape(group_offset, ())
  out = jnp.zeros((lhs.shape[0], rhs.shape[-1]), jnp.float32)
  for e in range(rhs.shape[0]):
    out = out + jnp.where((local == e)[:, None], lhs.astype(jnp.float32) @ rhs[e].astype(jnp.float32), 0.0)
  return out


@pytest.mark.cpu_only
class GmmSiluVjpTest(parameterized.TestCase):
  """ops.gmm_silu around stand-ins for the TPU kernel and gmm's TPU backward (see the module docstring)."""

  @parameterized.product(
      group_offset=(0, 1),
      # (use_dlhs_transpose_rhs, fuse_dlhs_scale, fuse_3d_dlhs_ps): each alone, so a swapped binding shows.
      dlhs_options=((False, False, False), (True, False, False), (False, True, False), (False, False, True)),
      heuristic=(False, True),
  )
  def test_custom_vjp(self, group_offset, dlhs_options, heuristic):
    dlhs_transpose, fuse_scale, fuse_3d = dlhs_options
    seen = {"tile_n": [], "return_pre_act": [], "bwd": []}

    def kernel_stand_in(
        lhs,
        rhs,
        group_sizes,
        tile_info,
        preferred_element_type,
        group_offset=None,
        fuse_act=None,
        return_pre_act=False,
        **kwargs,
    ):
      del kwargs
      self.assertEqual(fuse_act, "silu")
      seen["tile_n"].append(tile_info if callable(tile_info) else tile_info.tile_n)
      seen["return_pre_act"].append(return_pre_act)
      acc = _grouped_f32(lhs, rhs, group_sizes, 0 if group_offset is None else group_offset)
      n = acc.shape[-1] // 2
      act = (jax.nn.silu(acc[:, :n]) * acc[:, n:]).astype(preferred_element_type)
      if not return_pre_act:
        return act
      return act, acc[:, :n].astype(preferred_element_type), acc[:, n:].astype(preferred_element_type)

    def gmm_bwd_stand_in(**kwargs):  # keyword arguments only: ops._gmm_silu_bwd must name every one
      seen["bwd"].append(kwargs)
      lhs, rhs = kwargs["residual"][:2]
      return kwargs["grad"][0].astype(lhs.dtype), jnp.zeros_like(rhs), None, None, None, None, None

    k0, k1, k2 = jax.random.split(jax.random.key(group_offset), 3)
    lhs = jax.random.normal(k0, (_OP_M, _OP_K), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (len(_OP_GROUP_SIZES) - group_offset, _OP_K, 2 * _OP_N)) * 0.05).astype(jnp.bfloat16)
    cotangent = jax.random.normal(k2, (_OP_M, _OP_N), jnp.float32).astype(jnp.bfloat16)
    group_sizes, offset = jnp.asarray(_OP_GROUP_SIZES), jnp.array([group_offset], jnp.int32)

    def gmm_silu(l, r):
      return megablox_ops.gmm_silu(
          l,
          r,
          group_sizes,
          preferred_element_type=jnp.bfloat16,
          tiling=_OP_TILING,
          group_offset=offset,
          use_gmm_v2_heuristic_tiling=heuristic,
          use_dlhs_transpose_rhs=dlhs_transpose,
          fuse_dlhs_scale=fuse_scale,
          fuse_3d_dlhs_ps=fuse_3d,
      )

    with (
        mock.patch.object(gmm_v2, "gmm_v2", kernel_stand_in),
        mock.patch.object(megablox_ops, "_gmm_bwd", gmm_bwd_stand_in),
    ):
      gmm_silu(lhs, rhs)  # not differentiated: the activation alone
      _, vjp = jax.vjp(gmm_silu, lhs, rhs)
      dlhs, _ = vjp(cotangent)

    # The forward tile: tiling[2] halved (or gmm_v2's heuristic), every call; act alone undifferentiated, act + gate +
    # up differentiated.
    self.assertEqual(seen["tile_n"], [gmm_v2.calculate_tiling if heuristic else _OP_TILING[2] // 2] * 2)
    self.assertEqual(seen["return_pre_act"], [False, True])
    # The backward: gmm's backward gets [d gate | d up] of silu(gate) * up on the bf16 halves, bit for bit.
    acc = _grouped_f32(lhs, rhs, group_sizes, offset)
    gate, up = acc[:, :_OP_N].astype(jnp.bfloat16), acc[:, _OP_N:].astype(jnp.bfloat16)
    _, swiglu_vjp = jax.vjp(lambda g, u: (jax.nn.silu(g) * u).astype(g.dtype), gate, up)
    want = jnp.concatenate(swiglu_vjp(cotangent), axis=-1)
    np.testing.assert_array_equal(np.asarray(dlhs, np.float32), np.asarray(want, np.float32))
    self.assertLen(seen["bwd"], 1)
    bwd = seen["bwd"][0]
    np.testing.assert_array_equal(np.asarray(bwd["grad"][0], np.float32), np.asarray(want, np.float32))
    self.assertEqual(bwd["grad"][1:], (None, None))
    # Its residuals are this op's operands, and its options are this op's (the plain bf16 tokamax gmm_v2 GEMM).
    res_lhs, res_rhs, res_group_sizes, res_offset, *res_rest = bwd["residual"]
    for got, want_operand in ((res_lhs, lhs), (res_rhs, rhs), (res_group_sizes, group_sizes), (res_offset, offset)):
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want_operand))
    self.assertEqual(tuple(res_rest), (None, False, False, None))
    expected_options = {
        "lhs_dtype": jnp.bfloat16,
        "rhs_dtype": jnp.bfloat16,
        "preferred_element_type": jnp.bfloat16,
        "tiling": _OP_TILING,
        "transpose_rhs": False,
        "quantization_rule": None,
        "use_tokamax_backend": True,
        "use_gmm_v2": True,
        "use_gmm_v2_heuristic_tiling": heuristic,
        "return_rhs": False,
        "return_lhs": False,
        "fuse_dlhs_scale": fuse_scale,
        "use_dlhs_transpose_rhs": dlhs_transpose,
        "out_is_3d": False,
        "fuse_3d_dlhs_ps": fuse_3d,
        "tc_unsort_cfg": None,
    }
    self.assertEqual({name: bwd[name] for name in expected_options}, expected_options)

  def test_odd_forward_tile_is_rejected(self):
    lhs = jnp.zeros((_OP_M, _OP_K), jnp.bfloat16)
    rhs = jnp.zeros((len(_OP_GROUP_SIZES), _OP_K, 2 * _OP_N), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "must halve to a multiple of 128"):
      megablox_ops.gmm_silu(lhs, rhs, jnp.asarray(_OP_GROUP_SIZES), jnp.bfloat16, (64, 256, 192) + _OP_TILING[3:])


@pytest.mark.cpu_only
def test_fused_swiglu_wiring_on_cpu_mesh():
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


def _grouped_matmul(lhs, rhs, group_sizes, group_offset):
  """f32 lhs @ rhs[local expert of row]: rows of the groups [group_offset, group_offset + groups), others 0."""
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= ends[None, :], axis=1) - group_offset
  out = jnp.zeros((lhs.shape[0], rhs.shape[-1]), jnp.float32)
  for e in range(rhs.shape[0]):
    out = out + jnp.where((local == e)[:, None], lhs.astype(jnp.float32) @ rhs[e].astype(jnp.float32), 0.0)
  return out


def _gmm_stand_in(lhs, rhs, group_sizes, preferred_element_type=jnp.float32, tiling=None, group_offset=None, **kwargs):
  """jnp stand-in for megablox gmm (the TPU kernel), for the calls the layer makes here."""
  del tiling
  for name in ("partial_sum", "return_lhs", "return_rhs"):
    assert not kwargs.get(name), name
  offset = 0 if group_offset is None else group_offset
  return _grouped_matmul(lhs, rhs, group_sizes, offset).astype(preferred_element_type)


def _gmm_silu_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset=None, **kwargs):
  """megablox gmm_silu as the unfused composition of the gmm stand-in: silu(gate) * up of its rounded output."""
  del kwargs
  out = _gmm_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset)
  n = out.shape[-1] // 2
  return (jax.nn.silu(out[:, :n]) * out[:, n:]).astype(preferred_element_type)


def _halves_swapped_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset=None, **kwargs):
  """Deliberately wrong: treats the up half as the gate and the gate half as up."""
  del kwargs
  out = _gmm_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset)
  n = out.shape[-1] // 2
  return (jax.nn.silu(out[:, n:]) * out[:, :n]).astype(preferred_element_type)


def _offset_ignored_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, group_offset=None, **kwargs):
  """Deliberately wrong: ignores the group offset, i.e. computes the first shard's experts on every shard."""
  del group_offset
  return _gmm_silu_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, None, **kwargs)


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
  batch = int(cfg.per_device_batch_size * _NUM_DEVICES)
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32)
  token_mask = None
  if mode == "padmask":
    lens = jnp.array([_SEQ - 5 * (b + 1) for b in range(batch)])
    token_mask = (jnp.arange(_SEQ)[None, :] < lens[:, None]).astype(jnp.int32)
  return x, token_mask


def _run(cfg, mesh, params, x, token_mask, gmm_silu, gmm=None):
  """Loss, output and gradients with the stand-ins; the keyword arguments of every gmm_silu call traced, and the
  tiling of the plain path's up-projection gmm calls ([K, 2N] weight)."""
  silu_calls, wi_tilings = [], []
  gmm = gmm or _gmm_stand_in

  def recorded_gmm_silu(*args, **kwargs):
    silu_calls.append(kwargs)
    return gmm_silu(*args, **kwargs)

  def recorded_gmm(lhs, rhs, *args, **kwargs):
    if rhs.shape[-1] == 2 * cfg.moe_mlp_dim:
      wi_tilings.append(tuple(kwargs["tiling"]))
    return gmm(lhs, rhs, *args, **kwargs)

  with (
      mock.patch.object(megablox, "gmm", recorded_gmm),
      mock.patch.object(megablox, "gmm_silu", recorded_gmm_silu),
      jax.set_mesh(mesh),
      nn_partitioning.axis_rules(cfg.logical_axis_rules),
  ):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x, token_mask=token_mask)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    jax.make_jaxpr(loss_fn)(params, x)
    traced = list(silu_calls)
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), traced, sorted(set(wi_tilings))


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


def _case(label, overrides, mode="real", gmm_silu=_gmm_silu_stand_in):
  """The layer without and with the key on the same parameters; `gmm_silu` is the fused GMM's stand-in."""
  cfg_old = _cfg(**overrides)
  cfg_new = _cfg(**overrides, moe_wi_gmm_fused_swiglu=True)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, token_mask = _inputs(cfg_old, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, calls_old, wi_tilings = _run(cfg_old, mesh, params, x, token_mask, gmm_silu)
  new, calls_new, _ = _run(cfg_new, mesh, params, x, token_mask, gmm_silu)
  structure = not calls_old and len(calls_new) == cfg_new.num_moe_token_chunks
  # gmm_silu gets the tiling the plain path gives the up-projection gmm, and the config's DLHS options.
  want = {
      "tiling": wi_tilings[0] if len(wi_tilings) == 1 else wi_tilings,
      "use_gmm_v2_heuristic_tiling": cfg_new.use_gmm_v2_heuristic_tiling,
      "use_dlhs_transpose_rhs": cfg_new.moe_gmm_v2_dlhs_transpose_rhs,
      "fuse_dlhs_scale": cfg_new.moe_accumulate_wi_dlhs,
      "fuse_3d_dlhs_ps": cfg_new.moe_3d_dlhs_partial_sum,
      "preferred_element_type": jnp.bfloat16,
  }
  got = [{name: tuple(c[name]) if name == "tiling" else c[name] for name in want} for c in calls_new]
  bindings = bool(got) and all(g == want for g in got)
  print(
      f"{label}: [{'ok' if structure else 'FAIL'}] gmm_silu calls traced plain={len(calls_old)}"
      f" fused={len(calls_new)}; [{'ok' if bindings else 'FAIL'}] gmm_silu arguments {got[:1]} (want {want})",
      flush=True,
  )
  return _bit_identical(label, old, new) and structure and bindings


def _gmm_offset_ignored_stand_in(
    lhs, rhs, group_sizes, preferred_element_type=jnp.float32, tiling=None, group_offset=None, **kwargs
):
  """Deliberately wrong: a gmm that ignores the group offset, i.e. computes the first shard's experts everywhere."""
  del group_offset
  return _gmm_stand_in(lhs, rhs, group_sizes, preferred_element_type, tiling, None, **kwargs)


def _ragged_dropless_matches_unsorted(label, ragged_gmm=None):
  """Key off: the full-buffer (dropless) ring-of-experts ragged sort against the ring of experts without it.

  Both take their GMM routing operands from get_gmm_operands_for_local_experts, but only the ragged path uses a
  non-zero group offset on a full buffer. Not a comparison of the key: a reference the key does not touch.
  `ragged_gmm` replaces the gmm stand-in in the ragged run only.
  """
  cfg_ragged = _cfg(ragged_buffer_factor=-1.0)
  cfg_unsorted = _cfg(ragged_buffer_factor=-1.0, use_ragged_sort=False)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ragged), cfg_ragged.mesh_axes)
  x, token_mask = _inputs(cfg_ragged, "real")
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ragged.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_ragged, mesh), nnx.Param, ...)
  reference, _, _ = _run(cfg_unsorted, mesh, params, x, token_mask, _gmm_silu_stand_in)
  ragged, _, _ = _run(cfg_ragged, mesh, params, x, token_mask, _gmm_silu_stand_in, gmm=ragged_gmm)
  (loss_ref, out_ref, grads_ref), (loss_new, out_new, grads_new) = reference, ragged
  named = [("output", out_ref, out_new), ("loss", loss_ref, loss_new)] + [
      (jax.tree_util.keystr(path), a, b)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_ref)[0], jax.tree_util.tree_leaves(grads_new))
  ]
  passed, lines = True, []
  for name, a, b in named:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    rel = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
    ok = rel < _UNSORTED_REL_L2 and bool(np.isfinite(b).all()) and float(np.max(np.abs(a))) > 0
    passed &= ok
    lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: rel_l2 vs use_ragged_sort=False {rel:.2e}")
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


# The subset of the Qwen3.5-397B RL trainer benchmark's MoE settings that this file needs (the cases above do not
# all set them).
_BENCH_FLAGS = {
    "moe_chunk_pipeline": True,
    "moe_expert_weight_prefetch": "forward",
    "ring_of_experts_row_major_reduce_scatter": True,
    "moe_routing_maps": "device",
    "ragged_buffer_factor": 2.75,
    "fp8_moe_fake_quant": True,
    "moe_exclude_padding_from_routing": True,
    "moe_gmm_v2_dlhs_transpose_rhs": True,
}


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  pipelined = {"moe_chunk_pipeline": True}
  other_keys = {
      "moe_unsplit_prefused_wi": True,
      "moe_routing_index_math": "count",
      "moe_roe_packed_routing_gather": True,
  }
  results = {
      "plain_loop": _case("plain_loop", {}),
      "pipelined": _case("pipelined", pipelined),
      "dropless": _case("dropless", {"ragged_buffer_factor": -1.0}),
      "padding_excluded": _case("padding_excluded", {**pipelined, "moe_exclude_padding_from_routing": True}, "padmask"),
      "unsplit_wi": _case("unsplit_wi", {**pipelined, "moe_unsplit_prefused_wi": True}),
      "with_other_keys": _case("with_other_keys", {**pipelined, **other_keys}),
      "with_other_keys_bench_flags": _case("with_other_keys_bench_flags", {**_BENCH_FLAGS, **other_keys}, "padmask"),
      "key_off_ragged_dropless_matches_unsorted": _ragged_dropless_matches_unsorted(
          "key_off_ragged_dropless_matches_unsorted"
      ),
  }
  controls = {
      "control_halves_swapped": _case("control_halves_swapped", pipelined, gmm_silu=_halves_swapped_stand_in),
      "control_offset_ignored": _case(
          "control_offset_ignored", {"ragged_buffer_factor": -1.0}, gmm_silu=_offset_ignored_stand_in
      ),
      "control_key_off_ragged_offset_ignored": _ragged_dropless_matches_unsorted(
          "control_key_off_ragged_offset_ignored", ragged_gmm=_gmm_offset_ignored_stand_in
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

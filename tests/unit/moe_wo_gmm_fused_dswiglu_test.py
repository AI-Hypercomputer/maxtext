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
"""moe_wo_gmm_fused_dswiglu in the MoE layer, on CPU: the wiring of the fused SwiGLU backward, and the config validator.

With the key, the backward of the MoE SwiGLU runs in the epilogue of the down projection's (wo) DLHS GMM: the
tokamax gmm_v2 kernel with fuse_dact="swiglu" turns u = g @ wo^T into d[gate | up]. The up projection hands its
whole [gate | up] output (or, with moe_wi_gmm_fused_swiglu, the [gate | up] the fused wi GMM writes next to the
activation) to that epilogue, and the activation is passed on with its gradient stopped. With
moe_router_grad_from_wo_input the routing weight is the epilogue's row scale and the routing-weight gradient <u, h>
comes from the same epilogue.

The kernels are Pallas TPU kernels and do not run on CPU (their numerics are tested on TPU in
megablox_gmm_dswiglu_test.py, and the epilogue function itself on CPU there). Here the two Pallas entry points, gmm_v2
and tgmm_v2, are replaced by jnp stand-ins, so everything above them is the real code: moe.py, ragged_sort.py and
megablox ops.py (gmm's custom VJP, gmm_silu, gmm_dswiglu and gmm_dswiglu_bwd). The gmm_v2 stand-in is a grouped matmul
over the same group sizes and group offset, with the options the layer uses (fuse_act, return_pre_act, fuse_dact).

- With a stand-in epilogue that rounds where the unfused backward rounds (u to bf16, w * u to bf16, the bf16 SwiGLU
  VJP, the dot from the bf16 u and activation), the layer with the key reproduces the layer without it BIT FOR BIT:
  the output, the loss and every gradient. That checks everything the key changes in moe.py, ragged_sort.py and
  ops.py (which rows and routing operands reach the fused DLHS, the padding, the gradient routed through [gate | up]
  and not the activation, the routing weight as the row scale, the dot as the routing-weight gradient), not the
  kernel's numerics. Without the key the reference runs moe_prefused_swiglu_vjp, whose backward writes d[gate | up]
  in one pass as the epilogue does (split, an exact zero could flip sign in the add of the halves).
- With the kernel module's own epilogue (apply_dact_fn on the float32 accumulator, i.e. the fused numerics), the
  output and loss stay bit-identical and the gradients agree to bf16 rounding (relative L2 < 1e-2).

The layer is the shrunk Qwen3.5 routed MoE of moe_wi_gmm_fused_swiglu_test.py on 8 CPU devices (cp-as-ep, ragged sort,
local routing, two token chunks), in a subprocess so the forced device count applies. Cases: with and without
moe_router_grad_from_wo_input, with and without moe_wi_gmm_fused_swiglu, the plain and the pipelined loop, a dropless
buffer (group offsets not 0), padding excluded, and the measured stack's keys, with the bench's transposed-rhs DLHS
(moe_gmm_v2_dlhs_transpose_rhs: the stand-in honours transpose_rhs, so a wrong wo layout fails). Two-sided: the fused
DLHS must be traced exactly once per chunk with the key and never without it; the activation's VJP must never be built
with the key (its gradient is stopped, gate_up carries it) and must be with the unfused reference; and stand-in
epilogues that swap the gate and up halves, or that drop the row scale (the routing weight), must FAIL.

The gmm_v2 stand-in writes the fused wi GMM's [gate | up] (return_pre_act="concat") as the kernel does, one
[gate_j | up_j] block per forward n tile, and its fused DLHS reads it with the kernel module's own reader
(split_dact_gate_up) at the dact_pre_act_tile_n the layer passes. With wi_tile_fwd_mlp_dim = the MLP dim (two forward
n tiles per half, as at the 397B bench cell 1024 / 1024) the layer must pass the forward's tile_n and stay bit-identical
to the unfused layer; a reader that ignores the layout (the measured build) must FAIL there.

The validator cases also cover the kernels' tile rules, which it checks at config time when no mesh axis shards the
MoE MLP dim: wo_tile_dlhs_mlp_dim must be that dim, and with moe_wi_gmm_fused_swiglu wi_tile_fwd_mlp_dim must halve to
that dim or to a multiple of 128 dividing it (stack 9's 1024 against Qwen3.5-397B's 1024 is accepted).
"""

import functools
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
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 256
_MLP = 256
_SEQ = 64
_PASS_TOKEN = "MOE_WO_GMM_FUSED_DSWIGLU_CHECKS_PASSED"
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_NUM_DEVICES} --xla_allow_excess_precision=false"
_GRAD_REL_L2 = 1e-2  # The fused numerics: the gradients differ from the unfused ones by bf16 rounding.


def _cfg(**overrides):
  """The shrunk Qwen3.5 routed-MoE config with prefused weights and the tokamax gmm_v2 path the key requires."""
  kwargs = {
      "run_name": "moe_wo_gmm_fused_dswiglu",
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
      # One n tile spans the MLP dim in the wo DLHS, and each half in the fused wi forward (as the kernel requires).
      "wo_tile_dlhs_mlp_dim": _MLP,
      "wi_tile_fwd_mlp_dim": 2 * _MLP,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class FusedDswigluConfigTest(parameterized.TestCase):
  """The key is rejected where the down projection's fused gmm_v2 DLHS would not run or would not apply."""

  @parameterized.named_parameters(
      ("no_gmm_v2", {"use_gmm_v2": False}, "use_tokamax_gmm=True and use_gmm_v2=True"),
      ("not_prefused", {"prefuse_moe_weights": False}, "prefuse_moe_weights=True"),
      ("dense", {"sparse_matmul": False, "megablox": False}, "sparse_matmul=True"),
      ("gelu", {"mlp_activations": ["gelu", "linear"]}, "mlp_activations=['silu', 'linear']"),
      ("mlp_bias", {"mlp_bias": True}, "mlp_bias=False"),
      ("tc_ragged_sort", {"moe_tc_ragged_sort": True}, "moe_tc_ragged_sort=False"),
      ("chunk_wgrad", {"moe_accumulate_chunk_wgrad": True}, "moe_accumulate_chunk_wgrad=False"),
      ("saved_up_projection", {"remat_policy": "custom", "moe_mlpwi_1": "device"}, "saves it as moe_mlpwi_1"),
      ("minimal_remat", {"remat_policy": "minimal"}, "remat_policy=minimal saves it as mlpwi_0"),
      ("quantization", {"quantization": "int8"}, "no weight quantization"),
      ("clipped_swiglu", {"mlp_activations_limit": 7.0}, "mlp_activations_limit <= 0 (no clipping)"),
      # The kernels' one-n-tile rules, at config time (the MoE MLP dim is not sharded here, so it is _MLP per shard).
      ("dlhs_two_tiles", {"wo_tile_dlhs_mlp_dim": _MLP // 2}, f"wo_tile_dlhs_mlp_dim={_MLP}, the MoE MLP dim"),
      ("dlhs_tile_wider_than_mlp", {"wo_tile_dlhs_mlp_dim": 2 * _MLP}, f"wo_tile_dlhs_mlp_dim={_MLP}, the MoE MLP dim"),
      # With moe_wi_gmm_fused_swiglu the fused DLHS reads [gate | up] in pieces of gmm_silu's tile_n per half
      # (wi_tile_fwd_mlp_dim // 2): the MLP dim, or whole lane tiles of 128 that divide it.
      (
          "fused_wi_half_tile_off_the_lane_grid",  # 128 // 2 = 64 divides the MLP dim, but is not a lane tile
          {"moe_wi_gmm_fused_swiglu": True, "wi_tile_fwd_mlp_dim": 128},
          f"a wi_tile_fwd_mlp_dim that halves to {_MLP} or to a multiple of 128 dividing it",
      ),
      (
          "fused_wi_half_tile_not_dividing",  # 384 // 2 = 192
          {"moe_wi_gmm_fused_swiglu": True, "wi_tile_fwd_mlp_dim": 384},
          f"a wi_tile_fwd_mlp_dim that halves to {_MLP} or to a multiple of 128 dividing it",
      ),
      (
          "fused_wi_tile_odd",  # clipped to the MLP dim per half, but gmm_silu halves the tile itself
          {"moe_wi_gmm_fused_swiglu": True, "wi_tile_fwd_mlp_dim": 2 * _MLP + 1},
          f"a wi_tile_fwd_mlp_dim that halves to {_MLP} or to a multiple of 128 dividing it",
      ),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_wo_gmm_fused_dswiglu=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_wo_gmm_fused_dswiglu=True, **overrides)

  def test_prefused_swiglu_vjp_rejected_next_to_it(self):
    """With the key there is no separate activation backward left for moe_prefused_swiglu_vjp to fuse."""
    with self.assertRaisesRegex(ValueError, "the wo GMM's DLHS already writes"):
      _cfg(moe_wo_gmm_fused_dswiglu=True, moe_prefused_swiglu_vjp=True)

  @parameterized.named_parameters(
      ("alone", {}),
      ("with_fused_wi_swiglu", {"moe_wi_gmm_fused_swiglu": True}),
      # Stack 9: wi_tile_fwd_mlp_dim = the MLP dim, two forward n tiles per half, read in the per-tile layout.
      ("fused_wi_two_tiles_per_half", {"moe_wi_gmm_fused_swiglu": True, "wi_tile_fwd_mlp_dim": _MLP}),
      (
          "measured_stack",
          {
              "moe_chunk_pipeline": True,
              "moe_routing_index_math": "count",
              "moe_unsplit_prefused_wi": True,
              "moe_roe_packed_routing_gather": True,
              "moe_router_grad_from_wo_input": True,
              "moe_gmm_v2_dlhs_transpose_rhs": True,
          },
      ),
      # The wi forward tile only matters with moe_wi_gmm_fused_swiglu (stack 6 ran it at the MLP dim, unfused wi).
      ("unfused_wi_forward_tile_free", {"wi_tile_fwd_mlp_dim": _MLP}),
      # Heuristic tiling: calculate_tiling picks the tiles when traced, so the configured ones are not checked.
      (
          "heuristic_tiling",
          {
              "use_gmm_v2_heuristic_tiling": True,
              "moe_wi_gmm_fused_swiglu": True,
              "wi_tile_fwd_mlp_dim": _MLP,
              "wo_tile_dlhs_mlp_dim": _MLP // 2,
          },
      ),
      # A sharded MLP dim (here 2-way: _MLP / 2 per shard) is left to the kernel's check when traced.
      ("mlp_dim_sharded", {"ici_tensor_parallelism": 2, "wo_tile_dlhs_mlp_dim": _MLP // 2}),
  )
  def test_accepted(self, overrides):
    self.assertTrue(_cfg(moe_wo_gmm_fused_dswiglu=True, **overrides).moe_wo_gmm_fused_dswiglu)


@pytest.mark.cpu_only
def test_fused_dswiglu_wiring_on_cpu_mesh():
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


# ------------------------------------------------------------------------------------------------------------------
# jnp stand-ins for the two Pallas TPU kernels. Everything that calls them (ops.py, moe.py, ragged_sort.py) is real.
# ------------------------------------------------------------------------------------------------------------------
_TRACED = {}


def _grouped_matmul(lhs, rhs, group_sizes, group_offset):
  """f32 lhs @ rhs[local group of row] for the rows of the local groups, 0 for the other rows; and that row mask."""
  ends = jnp.cumsum(group_sizes)
  local = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= ends[None, :], axis=1) - jnp.reshape(group_offset, ())
  out = jnp.zeros((lhs.shape[0], rhs.shape[-1]), jnp.float32)
  for e in range(rhs.shape[0]):
    out = out + jnp.where((local == e)[:, None], lhs.astype(jnp.float32) @ rhs[e].astype(jnp.float32), 0.0)
  return out, (local >= 0) & (local < rhs.shape[0])


def _unfused_rounding_epilogue(acc, gate, up, row_scale, return_dot):
  """d[gate | up] rounded where the unfused backward rounds: the bf16 GMM output u, the bf16 w * u, the bf16 VJP."""
  u = acc.astype(jnp.bfloat16)
  u32 = u.astype(jnp.float32)
  dot = None
  if return_dot:
    act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    dot = jnp.sum(u32 * act.astype(jnp.float32), axis=-1, keepdims=True)
  du = u if row_scale is None else (u32 * row_scale).astype(jnp.bfloat16)
  _, vjp = jax.vjp(lambda g, v: (jax.nn.silu(g) * v).astype(jnp.bfloat16), gate, up)
  return jnp.concatenate(vjp(du), axis=-1), dot


def _kernel_epilogue(acc, gate, up, row_scale, return_dot):
  """The fused numerics: the kernel module's own epilogue on the float32 accumulator."""
  return gmm_v2.apply_dact_fn(acc, gate, up, row_scale, "swiglu", return_dot=return_dot)


def _halves_swapped_epilogue(acc, gate, up, row_scale, return_dot):
  """Deliberately wrong: takes the up half as the gate and the gate half as up."""
  return _unfused_rounding_epilogue(acc, up, gate, row_scale, return_dot)  # pylint: disable=arguments-out-of-order


def _row_scale_dropped_epilogue(acc, gate, up, row_scale, return_dot):
  """Deliberately wrong: ignores the per-row scale (with moe_router_grad_from_wo_input, the routing weight)."""
  del row_scale
  return _unfused_rounding_epilogue(acc, gate, up, None, return_dot)


def _concat_store(gate, up, tile_n):
  """The fused wi kernel's return_pre_act="concat" output: n tile j's accumulator [gate_j | up_j] is block j."""
  n = gate.shape[-1]
  blocks = [
      jnp.concatenate([gate[:, j * tile_n : (j + 1) * tile_n], up[:, j * tile_n : (j + 1) * tile_n]], -1)
      for j in range(n // tile_n)
  ]
  return jnp.concatenate(blocks, axis=-1)


def _legacy_reader(gate_up, pre_act_tile_n):
  """Deliberately wrong (the measured build): reads gate_up as the contiguous [gate | up] whatever its layout."""
  del pre_act_tile_n
  return gmm_v2.split_dact_gate_up(gate_up, None)


def _gmm_v2_stand_in(
    lhs,
    rhs,
    group_sizes,
    rhs_scale=None,
    rhs_bias=None,
    partial_sum=None,
    group_offset=None,
    *,
    tile_info=None,
    preferred_element_type=None,
    transpose_rhs=False,
    fuse_act=None,
    return_pre_act=False,
    fuse_dact=None,
    dact_gate_up=None,
    dact_row_scale=None,
    return_dact_dot=False,
    dact_pre_act_tile_n=None,
    epilogue=_unfused_rounding_epilogue,
    reader=None,
    **unused,
):
  """gmm_v2 for the calls the layer makes: rows outside the local groups are zero, as with zero_initialize."""
  del unused
  assert rhs_scale is None and rhs_bias is None and partial_sum is None
  out_dtype = preferred_element_type or lhs.dtype
  offset = 0 if group_offset is None else group_offset
  acc, rows = _grouped_matmul(lhs, rhs.swapaxes(1, 2) if transpose_rhs else rhs, group_sizes, offset)
  rows = rows[:, None]
  if fuse_dact is not None:
    _TRACED["fused_dlhs"] = _TRACED.get("fused_dlhs", 0) + 1
    n = acc.shape[-1]
    assert fuse_dact == "swiglu" and dact_gate_up.shape == (lhs.shape[0], 2 * n), (fuse_dact, dact_gate_up.shape)
    assert tile_info.tile_n == n, tile_info  # The kernel's single n tile.
    _TRACED.setdefault("pre_act_tile_n", set()).add(dact_pre_act_tile_n)
    scale = None if dact_row_scale is None else dact_row_scale.astype(jnp.float32).reshape(acc.shape[0], -1)[:, :1]
    # The kernel's reader of the layout (or a deliberately wrong one).
    gate, up = (reader or gmm_v2.split_dact_gate_up)(dact_gate_up, dact_pre_act_tile_n)
    d_gate_up, dot = epilogue(acc, gate, up, scale, return_dact_dot)
    d_gate_up = jnp.where(rows, d_gate_up, 0.0).astype(out_dtype)
    return (d_gate_up, jnp.where(rows[:, 0], dot[:, 0], 0.0)) if return_dact_dot else d_gate_up
  if fuse_act is not None:
    n = acc.shape[-1] // 2
    act = jnp.where(rows, jax.nn.silu(acc[:, :n]) * acc[:, n:], 0.0).astype(out_dtype)
    gate_up = jnp.where(rows, acc, 0.0).astype(out_dtype)
    if return_pre_act == "concat":
      _TRACED["fused_wi_concat"] = _TRACED.get("fused_wi_concat", 0) + 1
      _TRACED.setdefault("fwd_tile_n", set()).add(tile_info.tile_n)
      # [gate | up] is written one [gate_j | up_j] block per n tile, as the kernel writes it.
      return act, _concat_store(gate_up[:, :n], gate_up[:, n:], tile_info.tile_n)
    if return_pre_act:
      return act, gate_up[:, :n], gate_up[:, n:]
    return act
  return jnp.where(rows, acc, 0.0).astype(out_dtype)


def _tgmm_v2_stand_in(
    lhs,
    rhs,
    group_sizes,
    num_actual_groups,
    rhs_scale=None,
    partial_sum=None,
    preferred_element_type=None,
    group_offset=None,
    **unused,
):
  """tgmm_v2: per local group, lhs[rows of the group]^T @ rhs[rows of the group] in f32."""
  del unused
  assert rhs_scale is None and partial_sum is None
  ends = jnp.cumsum(group_sizes)
  offset = 0 if group_offset is None else group_offset
  local = jnp.sum(jnp.arange(lhs.shape[0])[:, None] >= ends[None, :], axis=1) - jnp.reshape(offset, ())
  outs = []
  for e in range(num_actual_groups):
    rows = (local == e)[:, None]
    outs.append(jnp.where(rows, lhs, 0).astype(jnp.float32).T @ jnp.where(rows, rhs, 0).astype(jnp.float32))
  return jnp.stack(outs).astype(preferred_element_type or rhs.dtype)


def _activation_vjp_tripwire(apply_ffn_activation):
  """RoutedMoE.apply_ffn_activation with an identity custom VJP on its output whose backward counts itself.

  _TRACED["activation_vjp"] counts how often JAX builds the activation's VJP. With the key the layer stops the
  activation's gradient (gate_up carries it), so it must stay 0: without that stop_gradient the fused backward's zero
  cotangent would run the whole SwiGLU VJP over [buffer, mlp] again, the very pass the key removes, with the same
  values. The unfused reference builds it (moe_prefused_swiglu_vjp's backward takes the activation's VJP). Values are
  unchanged either way.
  """

  @jax.custom_vjp
  def identity(y):
    return y

  def identity_fwd(y):
    return y, None

  def identity_bwd(_, cotangent):
    _TRACED["activation_vjp"] = _TRACED.get("activation_vjp", 0) + 1
    return (cotangent,)

  identity.defvjp(identity_fwd, identity_bwd)

  def wrapped(self, layer_w0, layer_w1):
    return identity(apply_ffn_activation(self, layer_w0, layer_w1))

  return wrapped


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


def _run(cfg, mesh, params, x, token_mask, epilogue, reader=None):
  """Loss, output and gradients with the stand-ins, and the fused kernel calls traced in the backward pass."""
  _TRACED.clear()
  with (
      mock.patch.object(gmm_v2, "gmm_v2", functools.partial(_gmm_v2_stand_in, epilogue=epilogue, reader=reader)),
      mock.patch.object(tgmm_v2, "tgmm_v2", _tgmm_v2_stand_in),
      mock.patch.object(
          moe.RoutedMoE, "apply_ffn_activation", _activation_vjp_tripwire(moe.RoutedMoE.apply_ffn_activation)
      ),
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

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    jax.make_jaxpr(grad_fn)(params, x)
    traced = dict(_TRACED)
    (loss, out), grads = jax.jit(grad_fn)(params, x)
  return jax.device_get((loss, out, grads)), traced


def _compare(label, old, new, exact, loose=()):
  """Output and loss must be bit-identical; gradients bit-identical (exact) or within _GRAD_REL_L2.

  Gradients whose path contains one of `loose` are only held to _GRAD_REL_L2 even when exact.
  """
  (loss_old, out_old, grads_old), (loss_new, out_new, grads_new) = old, new
  named = [("output", out_old, out_new, True), ("loss", loss_old, loss_new, True)]
  for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(grads_old)[0], jax.tree_util.tree_leaves(grads_new)):
    name = jax.tree_util.keystr(path)
    named.append((name, a, b, exact and not any(part in name for part in loose)))
  passed, lines = True, []
  for name, a, b, bitwise in named:
    a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
    rel = float(np.linalg.norm((a - b).ravel()) / max(float(np.linalg.norm(a.ravel())), 1e-30))
    ok = (bool(np.array_equal(a, b)) if bitwise else rel < _GRAD_REL_L2) and bool(np.isfinite(b).all())
    ok = ok and float(np.max(np.abs(a))) > 0
    passed &= ok
    lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: bit-identical={bool(np.array_equal(a, b))} rel_l2={rel:.3e}")
  print(f"{label}: {'PASS' if passed else 'FAIL'}", flush=True)
  print("\n".join(lines), flush=True)
  return passed


def _case(label, overrides, mode="real", epilogue=_unfused_rounding_epilogue, exact=True, reader=None):
  """The layer with the key against the layer without it (which runs the activation's VJP in one pass).

  With moe_router_grad_from_wo_input AND moe_wi_gmm_fused_swiglu, the unfused routing-weight gradient <u, h> reads
  the fused wi GMM's activation h, silu of the float32 accumulator, which the epilogue (it reads the bf16 gate and
  up) does not have: there the router's and the input's gradients are held to _GRAD_REL_L2, the rest stays exact.
  """
  cfg_new = _cfg(**overrides, moe_wo_gmm_fused_dswiglu=True)
  old_overrides = dict(overrides)
  if not overrides.get("moe_wi_gmm_fused_swiglu"):
    old_overrides["moe_prefused_swiglu_vjp"] = True
  cfg_old = _cfg(**old_overrides)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
  x, token_mask = _inputs(cfg_old, mode)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
    _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
  old, traced_old = _run(cfg_old, mesh, params, x, token_mask, epilogue)
  new, traced_new = _run(cfg_new, mesh, params, x, token_mask, epilogue, reader)
  chunks = cfg_new.num_moe_token_chunks
  want_concat = chunks if overrides.get("moe_wi_gmm_fused_swiglu") else 0
  # The layout the layer must hand the fused DLHS: the fused wi GMM's tile_n per half, or None for a GMM output.
  fwd_tile_n = min(cfg_new.wi_tile_fwd_mlp_dim, 2 * _MLP) // 2
  want_layout = {fwd_tile_n} if want_concat else {None}
  structure = (
      traced_old.get("fused_dlhs", 0) == 0
      and traced_new.get("fused_dlhs", 0) == chunks
      and traced_old.get("fused_wi_concat", 0) == 0
      and traced_new.get("fused_wi_concat", 0) == want_concat
      # With the key the activation's gradient is stopped; the tripwire fires on the reference's own activation VJP
      # (the reference with moe_wi_gmm_fused_swiglu has the activation inside the fused wi GMM, not here).
      and traced_new.get("activation_vjp", 0) == 0
      and (want_concat > 0 or traced_old.get("activation_vjp", 0) > 0)
      and traced_new.get("pre_act_tile_n") == want_layout
      and traced_new.get("fwd_tile_n", {fwd_tile_n}) == {fwd_tile_n}
  )
  print(f"{label}: [{'ok' if structure else 'FAIL'}] traced plain={traced_old} fused={traced_new}", flush=True)
  loose = ("['gate']", "[1]") if overrides.get("moe_router_grad_from_wo_input") and want_concat else ()
  return _compare(label, old, new, exact, loose) and structure


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.devices()
  pipelined = {"moe_chunk_pipeline": True}
  a14 = {"moe_router_grad_from_wo_input": True}
  silu = {"moe_wi_gmm_fused_swiglu": True}
  two_tiles = {"wi_tile_fwd_mlp_dim": _MLP}  # two forward n tiles per half, as at the 397B cell (1024 vs 1024)
  measured = {
      **pipelined,
      **a14,
      "moe_exclude_padding_from_routing": True,
      "moe_routing_index_math": "count",
      "moe_unsplit_prefused_wi": True,
      "moe_roe_packed_routing_gather": True,
      # As on the bench: the DLHS kernel reads wo as stored and transposes it itself (gmm_dswiglu_bwd's transpose_rhs
      # branch); the other cases hand it the explicit wo^T.
      "moe_gmm_v2_dlhs_transpose_rhs": True,
  }
  results = {
      # moe_router_grad_from_wo_input: the fused op of ring_ragged_unsort_down_projection.
      "a14_measured_stack": _case("a14_measured_stack", measured, "padmask"),
      "a14_measured_stack_fused_wi": _case("a14_measured_stack_fused_wi", {**measured, **silu}, "padmask"),
      "a14_plain_loop": _case("a14_plain_loop", a14),
      "a14_dropless_fused_wi": _case("a14_dropless_fused_wi", {**pipelined, **a14, **silu, "ragged_buffer_factor": -1.0}),
      # Without it: the down projection of the pipelined loop (_down_projection) and of the plain loop.
      "pipelined": _case("pipelined", {**pipelined, "moe_unsplit_prefused_wi": True}),
      "pipelined_fused_wi": _case("pipelined_fused_wi", {**pipelined, **silu}),
      "plain_loop_fused_wi": _case("plain_loop_fused_wi", silu),
      "plain_loop_dropless": _case("plain_loop_dropless", {"ragged_buffer_factor": -1.0}),
      # The fused numerics: the kernel's own epilogue.
      "kernel_epilogue_a14": _case("kernel_epilogue_a14", measured, "padmask", _kernel_epilogue, exact=False),
      "kernel_epilogue_fused_wi": _case(
          "kernel_epilogue_fused_wi", {**pipelined, **silu}, epilogue=_kernel_epilogue, exact=False
      ),
      # Two forward n tiles per half: the fused DLHS reads the [gate_j | up_j] layout.
      "a14_measured_stack_fused_wi_2tiles": _case(
          "a14_measured_stack_fused_wi_2tiles", {**measured, **silu, **two_tiles}, "padmask"
      ),
      "pipelined_fused_wi_2tiles": _case("pipelined_fused_wi_2tiles", {**pipelined, **silu, **two_tiles}),
      "kernel_epilogue_fused_wi_2tiles": _case(
          "kernel_epilogue_fused_wi_2tiles", {**pipelined, **silu, **two_tiles}, epilogue=_kernel_epilogue, exact=False
      ),
  }
  controls = {
      "control_halves_swapped": _case("control_halves_swapped", pipelined, epilogue=_halves_swapped_epilogue),
      "control_halves_swapped_fused_wi": _case(
          "control_halves_swapped_fused_wi", {**pipelined, **silu}, epilogue=_halves_swapped_epilogue, exact=False
      ),
      "control_row_scale_dropped": _case(
          "control_row_scale_dropped", measured, "padmask", _row_scale_dropped_epilogue, exact=False
      ),
      # The measured build's reader at two forward tiles per half: gate = [gate_0 | up_0].
      "control_legacy_reader_fused_wi_2tiles": _case(
          "control_legacy_reader_fused_wi_2tiles",
          {**pipelined, **silu, **two_tiles},
          exact=False,
          reader=_legacy_reader,
      ),
      "control_legacy_reader_a14_fused_wi_2tiles": _case(
          "control_legacy_reader_a14_fused_wi_2tiles",
          {**measured, **silu, **two_tiles},
          "padmask",
          exact=False,
          reader=_legacy_reader,
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

# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and HLO parity tests for `moe_ring_pipeline_mode` and `ring_of_experts_row_major_reduce_scatter`."""

import collections
import functools
import os
import re
import subprocess
import sys
import types
from unittest import mock

from absl import app
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental.layout import Layout, with_layout_constraint
from jax.extend import core as jex_core
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers import nnx_decoders
from maxtext.layers.initializers import nd_dense_init
from maxtext.models import qwen3
from maxtext.models import qwen3_5
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_post_train_test_config_path, get_test_config_path

_NUM_EXPERTS = 8
_TOP_K = 2
_EMB = 128
_MLP = 128
_SEQ = 32
_SUBPROC_PASS_TOKEN = "MOE_RING_PIPELINE_8DEV_BIT_EXACT_PASSED"
_SUBPROC_ENV = "_MOE_RING_PIPELINE_8DEV_SUBPROC"
_BARRIER_TIE = moe.barrier_tie
_COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")


def _subjaxprs(eqn):
  """Yields jaxprs nested inside an equation's parameters."""
  for param in eqn.params.values():
    for sub in param if isinstance(param, (list, tuple)) else (param,):
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      if hasattr(sub, "eqns"):
        yield sub


def _count_primitive(jaxpr, name: str) -> int:
  """Counts equations with `primitive.name == name`, including nested jaxprs."""
  return sum(
      (eqn.primitive.name == name) + sum(_count_primitive(sub, name) for sub in _subjaxprs(eqn)) for eqn in jaxpr.eqns
  )


def _count_checkpoint_name(jaxpr, target_name: str) -> int:
  """Counts `name` checkpoint equations matching `target_name`, including nested jaxprs."""
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "name" and eqn.params.get("name") == target_name:
      total += 1
    for sub in _subjaxprs(eqn):
      total += _count_checkpoint_name(sub, target_name)
  return total


def _layouts(jaxpr):
  """Returns `(major_to_minor, rank)` for every `layout_constraint` equation, including nested jaxprs."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "layout_constraint":
      found.append((tuple(eqn.params["layout"].major_to_minor), eqn.invars[0].aval.ndim))
    for sub in _subjaxprs(eqn):
      found.extend(_layouts(sub))
  return found


def _is_barrier(eqn) -> bool:
  """True for `optimization_barrier` or a `custom_vjp_call` whose body is a single `optimization_barrier`."""
  if eqn.primitive.name == "optimization_barrier":
    return True
  if not eqn.primitive.name.startswith("custom_vjp_call"):
    return False
  body = next(_subjaxprs(eqn), None)
  return body is not None and len(body.eqns) == 1 and body.eqns[0].primitive.name == "optimization_barrier"


def _unread_barrier_outputs(jaxpr):
  """Returns unread barrier output descriptors across `jaxpr` and nested sub-jaxprs."""
  unread = []
  read = {v for eqn in jaxpr.eqns for v in eqn.invars if not isinstance(v, jex_core.Literal)}
  read |= {v for v in jaxpr.outvars if not isinstance(v, jex_core.Literal)}
  for k, eqn in enumerate(jaxpr.eqns):
    if _is_barrier(eqn):
      unread.extend(f"barrier {k} output {i}" for i, v in enumerate(eqn.outvars) if v not in read)
    else:
      for sub in _subjaxprs(eqn):
        unread.extend(_unread_barrier_outputs(sub))
  return unread


def _split_hlo_shape(rest: str) -> tuple[str, str]:
  """Splits `'shape opcode(...)'` at the end of the result shape."""
  if not rest.startswith("("):
    shape, _, tail = rest.partition(" ")
    return shape, tail
  depth = 0
  for i, char in enumerate(rest):
    depth += (char == "(") - (char == ")")
    if depth == 0:
      return rest[: i + 1], rest[i + 1 :].lstrip()
  return rest, ""


def _collectives(hlo_text: str) -> collections.Counter:
  """Returns multiset of `(opcode, shape_without_layout)` of collectives in compiled HLO."""
  found: collections.Counter = collections.Counter()
  for line in hlo_text.splitlines():
    if " = " not in line:
      continue
    shape, rest = _split_hlo_shape(line.split(" = ", 1)[1])
    op = rest.split("(", 1)[0].removesuffix("-start")
    if op in _COLLECTIVES:
      found[(op, re.sub(r"\{[^}]*\}", "", shape))] += 1
  return found


def _count_hlo_opt_barriers(hlo_text: str) -> int:
  """Returns the count of `opt-barrier(` instructions in pre-optimization HLO text."""
  return sum(1 for line in hlo_text.splitlines() if "opt-barrier(" in line)


def _make_mesh(cfg, num_devices: int) -> Mesh:
  """Builds a `Mesh` over the first `num_devices` JAX devices matching `cfg`."""
  devices = jax.devices()[:num_devices]
  ici_sizes = [int(sz) for sz in cfg.ici_parallelism]
  used = 1
  neg_idx = None
  for i, sz in enumerate(ici_sizes):
    if sz == -1:
      neg_idx = i
    else:
      used *= sz
  if neg_idx is not None:
    ici_sizes[neg_idx] = num_devices // used
  elif math.prod(ici_sizes) != num_devices:
    # Default fsdp absorbs remaining devices if all axes defaulted to 1.
    fsdp_idx = list(cfg.mesh_axes).index("fsdp")
    ici_sizes[fsdp_idx] = num_devices // math.prod(ici_sizes)
  arr = np.asarray(devices).reshape(ici_sizes)
  return Mesh(arr, cfg.mesh_axes)


import math  # pylint: disable=wrong-import-position,wrong-import-order


def _moe_cfg(run_suffix: str, num_devices: int = 4, **overrides):
  """Creates a shrunk DeepSeek/Qwen3.5 routed-MoE config for Ring-of-Experts testing."""
  kwargs = {
      "run_name": f"moe_ring_pipeline_{run_suffix}",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": True,
      "base_num_decoder_layers": 1,
      "num_experts": _NUM_EXPERTS,
      "num_experts_per_tok": _TOP_K,
      "base_emb_dim": _EMB,
      "base_mlp_dim": _MLP,
      "base_moe_mlp_dim": _MLP,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": max(1.0, 4.0 / num_devices),
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "float32_gate_logits": True,
      "megablox": False,
      "sparse_matmul": True,
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "ici_fsdp_parallelism": 1,
      "ici_context_parallelism": 1,
      "ici_tensor_parallelism": 1,
      "ici_expert_parallelism": num_devices,
      "context_parallel_load_balance": False,
      "moe_ring_pipeline_mode": "none",
      "ring_of_experts_row_major_reduce_scatter": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize(
      [os.path.join(MAXTEXT_PKG_DIR, "trainers", "pre_train", "train.py"), get_test_config_path()], **kwargs
  )


def _build_moe(cfg, mesh: Mesh, force_mode: str | None = None):
  """Builds a `RoutedMoE` instance on `mesh`."""
  model = moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed_router", None),
      intermediate_dim=cfg.moe_mlp_dim,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      rngs=nnx.Rngs(params=0),
  )
  if force_mode is not None:
    model.moe_ring_pipeline_mode = force_mode
  return model


def _make_inputs(cfg, num_devices: int, *, forced_mode: bool = False):
  """Builds deterministic input activations and optional forced expert assignments."""
  batch = max(1, int(round(cfg.per_device_batch_size * num_devices)))
  in_dtype = jnp.float32 if cfg.dtype == jnp.float32 else jnp.bfloat16
  x = jax.random.normal(jax.random.PRNGKey(7), (batch, _SEQ, _EMB), jnp.float32).astype(in_dtype)
  upstream = jax.random.normal(jax.random.PRNGKey(13), (batch, _SEQ, _EMB), jnp.float32)
  if not forced_mode:
    return x, upstream, None
  keys = jax.random.split(jax.random.PRNGKey(11), batch * _SEQ)
  ids = jax.vmap(lambda k: jax.random.permutation(k, _NUM_EXPERTS)[:_TOP_K])(keys).reshape(batch, _SEQ, _TOP_K)
  pos = jnp.arange(_SEQ)[None, :, None]
  ids = jnp.where((pos % 5 == 0) & (jnp.arange(_TOP_K) == _TOP_K - 1), -1, ids)
  ids = jnp.where((pos % 7 == 3) & (jnp.arange(_TOP_K) == 1), ids[..., :1], ids)
  return x, upstream, ids.astype(jnp.int32)


def _run_layer_and_grads(cfg, mesh: Mesh, x, upstream, forced=None, shared_params=None, force_mode: str | None = None):
  """Runs forward + backward wrt `params` (`gate`, `wi_0`, `wi_1`, `wo`) and `x`, returning host values and diagnostics."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = _build_moe(cfg, mesh, force_mode=force_mode)
    graphdef, init_params, rest = nnx.split(model, nnx.Param, ...)
    params = init_params if shared_params is None else shared_params

    def loss_fn(p, act):
      m = nnx.merge(graphdef, p, rest)
      out, lb_loss, bias_updates = m(act, forced_routed_experts=forced)
      loss = jnp.mean(out.astype(jnp.float32) * upstream)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, (out, lb_loss, bias_updates)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    fwd_jaxpr = jax.make_jaxpr(loss_fn)(params, x).jaxpr
    grad_jaxpr = jax.make_jaxpr(grad_fn)(params, x).jaxpr
    fwd_hlo = jax.jit(loss_fn).lower(params, x).as_text("hlo")
    grad_hlo = jax.jit(grad_fn).lower(params, x).as_text("hlo")
    (loss, aux), (param_grads, x_grad) = jax.jit(grad_fn)(params, x)

    # Isolate pure VJP pullback HLO to prove zero opt-barrier in backward pass.
    _, vjp_fn = jax.vjp(lambda p, act: loss_fn(p, act)[0], params, x)
    bwd_only_hlo = jax.jit(vjp_fn).lower(jnp.ones((), dtype=jnp.float32)).as_text("hlo")

  layouts = _layouts(fwd_jaxpr)
  fwd_barriers = _count_hlo_opt_barriers(fwd_hlo)
  grad_total_barriers = _count_hlo_opt_barriers(grad_hlo)
  bwd_only_barriers = _count_hlo_opt_barriers(bwd_only_hlo)
  diag = {
      "fwd_jaxpr_barriers": _count_primitive(fwd_jaxpr, "optimization_barrier"),
      "fwd_hlo_opt_barriers": fwd_barriers,
      "grad_hlo_total_opt_barriers": grad_total_barriers,
      "vjp_pullback_opt_barriers": bwd_only_barriers,
      "combine_order_names": _count_checkpoint_name(fwd_jaxpr, "moe_combine_order"),
      "layout_constraints": len(layouts),
      "not_row_major": [m for m, ndim in layouts if m != tuple(range(ndim))],
      "unread_barriers": _unread_barrier_outputs(fwd_jaxpr) + _unread_barrier_outputs(grad_jaxpr),
      "fwd_hlo": fwd_hlo,
  }
  out, lb_loss, bias_updates = aux
  host_vals = jax.device_get((loss, out, lb_loss, bias_updates, param_grads, x_grad))
  return host_vals, diag, params


class HelperPrimitivesTest(absltest.TestCase):
  """Direct unit tests for `_forward_optimization_barrier`, `barrier_tie`, `forward_row_major`, and `_order_after`."""

  def test_forward_optimization_barrier_has_zero_backward_opt_barriers(self):
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    y = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4) * 0.5

    def fwd_only_fn(a, b):
      a, b = moe.barrier_tie(a, b, backward=False)
      return jnp.sum(a * 2.0 + b * 3.0)

    def fwd_bwd_fn(a, b):
      a, b = moe.barrier_tie(a, b, backward=True)
      return jnp.sum(a * 2.0 + b * 3.0)

    ga, gb = jax.grad(fwd_only_fn, argnums=(0, 1))(x, y)
    np.testing.assert_array_equal(ga, jnp.full_like(x, 2.0))
    np.testing.assert_array_equal(gb, jnp.full_like(y, 3.0))

    fwd_hlo = jax.jit(fwd_only_fn).lower(x, y).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(fwd_hlo), 1)

    _, vjp_fwd_only = jax.vjp(fwd_only_fn, x, y)
    vjp_hlo = jax.jit(vjp_fwd_only).lower(jnp.ones((), jnp.float32)).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(vjp_hlo), 0)

    grad_fwd_only_hlo = jax.jit(jax.grad(fwd_only_fn, argnums=(0, 1))).lower(x, y).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(grad_fwd_only_hlo), 0)
    val_and_grad_fwd_only_hlo = jax.jit(jax.value_and_grad(fwd_only_fn, argnums=(0, 1))).lower(x, y).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(val_and_grad_fwd_only_hlo), 1)

    # Negative control: backward=True emits a transposed opt-barrier in backward!
    _, vjp_both = jax.vjp(fwd_bwd_fn, x, y)
    vjp_both_hlo = jax.jit(vjp_both).lower(jnp.ones((), jnp.float32)).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(vjp_both_hlo), 1)
    grad_both_hlo = jax.jit(jax.grad(fwd_bwd_fn, argnums=(0, 1))).lower(x, y).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(grad_both_hlo), 1)
    val_and_grad_both_hlo = jax.jit(jax.value_and_grad(fwd_bwd_fn, argnums=(0, 1))).lower(x, y).as_text("hlo")
    self.assertEqual(_count_hlo_opt_barriers(val_and_grad_both_hlo), 2)

  def test_forward_row_major_constrains_forward_only(self):
    x = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4)
    # Eager (non-tracer) calls must return x unchanged without invoking with_layout_constraint.
    np.testing.assert_array_equal(max_utils.forward_row_major(x), x)
    np.testing.assert_array_equal(max_utils.forward_layout_constraint(x, (2, 0, 1)), x)
    np.testing.assert_array_equal(
        jax.grad(lambda a: jnp.sum(max_utils.forward_row_major(a) * 3.0))(x), jnp.full_like(x, 3.0)
    )
    np.testing.assert_array_equal(
        jax.grad(lambda a: jnp.sum(max_utils.forward_layout_constraint(a, (2, 0, 1)) * 3.0))(x), jnp.full_like(x, 3.0)
    )

    def count_layouts_in_grad(f):
      return len(_layouts(jax.make_jaxpr(jax.grad(lambda a: jnp.sum(f(a) * 3.0)))(x).jaxpr))

    self.assertEqual(count_layouts_in_grad(max_utils.forward_row_major), 1)
    self.assertEqual(count_layouts_in_grad(lambda a: max_utils.forward_layout_constraint(a, (2, 0, 1))), 1)
    self.assertEqual(count_layouts_in_grad(lambda a: with_layout_constraint(a, Layout(major_to_minor=(0, 1, 2)))), 2)

    # Verify qwen3._pin_gdn_gated_norm_layout reuses shared max_utils forward-only helpers cleanly.
    x4 = jnp.arange(48.0, dtype=jnp.float32).reshape(2, 3, 2, 4)
    np.testing.assert_array_equal(qwen3._pin_gdn_gated_norm_layout(x4, "token_minor"), x4)  # pylint: disable=protected-access
    np.testing.assert_array_equal(qwen3._pin_gdn_gated_norm_layout(x4, "natural"), x4)  # pylint: disable=protected-access
    tm_layouts = _layouts(
        jax.make_jaxpr(jax.grad(lambda a: jnp.sum(qwen3._pin_gdn_gated_norm_layout(a, "token_minor") * 2.0)))(x4).jaxpr  # pylint: disable=protected-access
    )
    nat_layouts = _layouts(
        jax.make_jaxpr(jax.grad(lambda a: jnp.sum(qwen3._pin_gdn_gated_norm_layout(a, "natural") * 2.0)))(x4).jaxpr  # pylint: disable=protected-access
    )
    self.assertEqual(tm_layouts, [((0, 2, 3, 1), 4)])
    self.assertEqual(nat_layouts, [((0, 1, 2, 3), 4)])

  def test_order_after_preserves_values_and_cotangents(self):
    v = jax.random.normal(jax.random.PRNGKey(1), (16, 4), jnp.float32)
    d = jax.random.normal(jax.random.PRNGKey(2), (4, 8, 16), jnp.bfloat16)
    np.testing.assert_array_equal(moe._order_after(v, d), v)  # pylint: disable=protected-access
    gv, gd = jax.grad(lambda a, b: jnp.sum(moe._order_after(a, b) * 3.0), argnums=(0, 1))(v, d)  # pylint: disable=protected-access
    np.testing.assert_array_equal(gv, jnp.full_like(v, 3.0))
    np.testing.assert_array_equal(gd, jnp.zeros_like(d))

    with self.assertRaisesRegex(ValueError, "floating-point"):
      moe._order_after(jnp.ones((4,), dtype=jnp.int32), d)  # pylint: disable=protected-access


class MoeRingPipelineConfigTest(parameterized.TestCase):
  """Config parsing, `'none'` normalization, and validator tests across `MaxTextConfig` and `RLConfig`."""

  def _init_pretrain(self, **kwargs):
    return pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "trainers", "pre_train", "train.py"), get_test_config_path()],
        run_name="moe_ring_pipeline_cfg",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        compile_topology="tpu7x-8",
        compile_topology_num_slices=1,
        **kwargs,
    )

  def _init_rl(self, **kwargs):
    return pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "trainers", "post_train", "rl", "train_rl.py"), get_post_train_test_config_path()],
        run_name="moe_ring_pipeline_rl_cfg",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        compile_topology="tpu7x-8",
        compile_topology_num_slices=1,
        **kwargs,
    )

  def test_defaults_and_none_normalization(self):
    cfg = self._init_pretrain()
    self.assertEqual(cfg.moe_ring_pipeline_mode, "none")
    self.assertFalse(cfg.ring_of_experts_row_major_reduce_scatter)

    for raw in ("none", "None", None):
      cfg_norm = self._init_pretrain(moe_ring_pipeline_mode=raw)
      self.assertEqual(cfg_norm.moe_ring_pipeline_mode, "none")

  @parameterized.named_parameters(
      ("combine_per_chunk", "combine_per_chunk"),
      ("software_pipeline", "software_pipeline"),
  )
  def test_valid_modes_in_pretrain_and_rl(self, mode):
    common = {
        "model_name": "deepseek3-test",
        "override_model_config": True,
        "ici_expert_parallelism": 2,
        "use_ring_of_experts": True,
        "use_ragged_sort": True,
        "num_moe_token_chunks": 2,
        "moe_ring_pipeline_mode": mode,
        "ring_of_experts_row_major_reduce_scatter": True,
        "remat_policy": "custom",
    }
    cfg = self._init_pretrain(**common)
    self.assertEqual(cfg.moe_ring_pipeline_mode, mode)
    self.assertTrue(cfg.ring_of_experts_row_major_reduce_scatter)
    self.assertNotIn("moe_combine_order", cfg.tensors_on_device)
    self.assertEqual("moe_combine_order" in maxtext_utils.get_save_and_offload_names(cfg)[0], mode == "combine_per_chunk")

    rl_cfg = self._init_rl(**common)
    self.assertEqual(rl_cfg.moe_ring_pipeline_mode, mode)
    self.assertTrue(rl_cfg.ring_of_experts_row_major_reduce_scatter)
    self.assertNotIn("moe_combine_order", rl_cfg.tensors_on_device)
    self.assertEqual(
        "moe_combine_order" in maxtext_utils.get_save_and_offload_names(rl_cfg)[0], mode == "combine_per_chunk"
    )

    cfg_cbwd = self._init_pretrain(**common, moe_quantize_combine_bwd_method="rowwise")
    rl_cbwd = self._init_rl(**common, moe_quantize_combine_bwd_method="rowwise")
    self.assertEqual(cfg_cbwd.moe_quantize_combine_bwd_method, "rowwise")
    self.assertEqual(rl_cbwd.moe_quantize_combine_bwd_method, "rowwise")

  @parameterized.named_parameters(
      (
          "row_major_without_roe",
          {"ring_of_experts_row_major_reduce_scatter": True},
          "use_ring_of_experts=True",
      ),
      (
          "row_major_dense_matmul",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "sparse_matmul": False,
              "ring_of_experts_row_major_reduce_scatter": True,
          },
          "sparse_matmul=True",
      ),
      (
          "pipeline_without_roe",
          {"moe_ring_pipeline_mode": "software_pipeline", "num_moe_token_chunks": 2},
          "use_ring_of_experts=True",
      ),
      (
          "combine_per_chunk_single_chunk",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 1,
              "moe_ring_pipeline_mode": "combine_per_chunk",
          },
          "num_moe_token_chunks > 1",
      ),
      (
          "software_pipeline_single_chunk",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 1,
              "moe_ring_pipeline_mode": "software_pipeline",
          },
          "num_moe_token_chunks > 1",
      ),
      (
          "software_pipeline_with_chunk_barrier",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 2,
              "moe_chunk_barrier": True,
              "moe_ring_pipeline_mode": "software_pipeline",
          },
          "moe_chunk_barrier=False",
      ),
      (
          "software_pipeline_with_emb_chunks",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "use_gmm_v2": True,
              "num_moe_token_chunks": 2,
              "num_moe_emb_chunks": 2,
              "moe_ring_pipeline_mode": "software_pipeline",
          },
          "num_moe_emb_chunks <= 0",
      ),
      (
          "pipeline_with_tc_weights_on_activation",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "use_ragged_sort": True,
              "ragged_buffer_factor": 1.5,
              "moe_tc_ragged_sort": True,
              "moe_tc_ragged_weights_on_activation": True,
              "num_moe_token_chunks": 2,
              "moe_ring_pipeline_mode": "combine_per_chunk",
          },
          "moe_tc_ragged_weights_on_activation=False",
      ),
      (
          "software_pipeline_with_qa2a_dispatch_fwd",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 2,
              "moe_ring_pipeline_mode": "software_pipeline",
              "moe_quantize_dispatch_fwd_method": "rowwise",
          },
          "quantizes the EP ragged all-to-all and requires use_ring_of_experts=False",
      ),
      (
          "combine_per_chunk_with_qa2a_combine_fwd",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 2,
              "moe_ring_pipeline_mode": "combine_per_chunk",
              "moe_quantize_combine_fwd_method": "rowwise",
          },
          "quantizes the EP ragged all-to-all and requires use_ring_of_experts=False",
      ),
      (
          "row_major_rs_with_qa2a_dispatch_bwd",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "ring_of_experts_row_major_reduce_scatter": True,
              "moe_quantize_dispatch_bwd_method": "rowwise",
          },
          "quantizes the EP ragged all-to-all and requires use_ring_of_experts=False",
      ),
      (
          "software_pipeline_with_qa2a_combine_bwd_format_suffix",
          {
              "ici_expert_parallelism": 2,
              "use_ring_of_experts": True,
              "num_moe_token_chunks": 2,
              "moe_ring_pipeline_mode": "software_pipeline",
              "moe_quantize_combine_bwd_method": "rowwise@e4m3",
          },
          "with use_ring_of_experts=True",
      ),
  )
  def test_invalid_combinations_rejected_in_pretrain_and_rl(self, kwargs, expected_msg):
    base = {"model_name": "deepseek3-test", "override_model_config": True}
    with self.assertRaisesRegex(ValueError, expected_msg):
      self._init_pretrain(**base, **kwargs)
    with self.assertRaisesRegex(ValueError, expected_msg):
      self._init_rl(**base, **kwargs)


class MoeRingPipelineParityAndHloTest(parameterized.TestCase):
  """Multi-device numerical parity, HLO `opt-barrier`, layout-constraint, and remat collective tests."""

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(jax.devices())}.")

  def _assert_parity_against_baseline(
      self, num_devices: int, common_overrides: dict, test_overrides: dict, rtol: float = 2e-3, atol: float = 2e-3
  ):
    """Asserts forward output, loss, and all input/weight gradients match the baseline `'none'` mode."""
    cfg_ref = _moe_cfg("ref", num_devices=num_devices, **common_overrides)
    cfg_new = _moe_cfg("new", num_devices=num_devices, **{**common_overrides, **test_overrides})
    mesh = _make_mesh(cfg_ref, num_devices)
    forced_mode = bool(common_overrides.get("_forced_mode", False))
    x, upstream, forced = _make_inputs(cfg_ref, num_devices, forced_mode=forced_mode)

    ref_vals, ref_diag, shared_params = _run_layer_and_grads(cfg_ref, mesh, x, upstream, forced=forced)
    new_vals, new_diag, _ = _run_layer_and_grads(cfg_new, mesh, x, upstream, forced=forced, shared_params=shared_params)

    loss_ref, out_ref, lb_ref, bu_ref, pgrads_ref, xgrad_ref = ref_vals
    loss_new, out_new, lb_new, bu_new, pgrads_new, xgrad_new = new_vals

    np.testing.assert_allclose(np.asarray(out_new, np.float32), np.asarray(out_ref, np.float32), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(loss_new, np.float32), np.asarray(loss_ref, np.float32), rtol=rtol, atol=atol)
    if lb_ref is not None:
      np.testing.assert_allclose(np.asarray(lb_new, np.float32), np.asarray(lb_ref, np.float32), rtol=rtol, atol=atol)
    if bu_ref is not None:
      np.testing.assert_array_equal(np.asarray(bu_new), np.asarray(bu_ref))

    # Check x gradient AND every parameter gradient leaf (gate, wi_0, wi_1, wo).
    np.testing.assert_allclose(np.asarray(xgrad_new, np.float32), np.asarray(xgrad_ref, np.float32), rtol=rtol, atol=atol)
    ref_leaves = jax.tree_util.tree_flatten_with_path(pgrads_ref)[0]
    new_leaves = jax.tree_util.tree_leaves(pgrads_new)
    self.assertGreaterEqual(len(ref_leaves), 4)
    for (path, g_ref), g_new in zip(ref_leaves, new_leaves, strict=True):
      g_ref_f32 = np.asarray(g_ref, np.float32)
      g_new_f32 = np.asarray(g_new, np.float32)
      self.assertGreater(float(np.max(np.abs(g_ref_f32))), 0.0, msg=f"Zero ref grad for {jax.tree_util.keystr(path)}")
      np.testing.assert_allclose(g_new_f32, g_ref_f32, rtol=rtol, atol=atol, err_msg=jax.tree_util.keystr(path))

    return ref_diag, new_diag

  @parameterized.named_parameters(
      (
          "combine_per_chunk_ep4_ragged_f32gate",
          {"dtype": "bfloat16", "float32_gate_logits": True, "ici_expert_parallelism": 4},
          {"moe_ring_pipeline_mode": "combine_per_chunk", "ring_of_experts_row_major_reduce_scatter": False},
          5e-3,
          2e-3,
      ),
      (
          "software_pipeline_ep4_f32gate",
          {
              "dtype": "float32",
              "scan_layers": False,
              "float32_gate_logits": True,
              "ici_expert_parallelism": 4,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": False},
          1e-5,
          1e-5,
      ),
      (
          "software_pipeline_row_major_ep4_bf16gate",
          {
              "dtype": "bfloat16",
              "float32_gate_logits": False,
              "ici_expert_parallelism": 4,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": True},
          5e-3,
          2e-3,
      ),
      (
          "combine_per_chunk_row_major_fsdp2_ep2",
          {
              "dtype": "float32",
              "scan_layers": False,
              "ici_fsdp_parallelism": 2,
              "ici_expert_parallelism": 2,
              "load_balance_loss_weight": 0.01,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "combine_per_chunk", "ring_of_experts_row_major_reduce_scatter": True},
          1e-5,
          1e-5,
      ),
      (
          "software_pipeline_row_major_fsdp2_ep2_4chunks",
          {
              "dtype": "bfloat16",
              "ici_fsdp_parallelism": 2,
              "ici_expert_parallelism": 2,
              "num_moe_token_chunks": 4,
              "moe_topk_before_ep_all_gather": True,
              "moe_routing_index_math": "count",
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": True},
          5e-3,
          2e-3,
      ),
      (
          "software_pipeline_tp2_ep2_f32",
          {
              "dtype": "float32",
              "scan_layers": False,
              "ici_tensor_parallelism": 2,
              "ici_expert_parallelism": 2,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": True},
          1e-5,
          1e-5,
      ),
      (
          "software_pipeline_cp_as_ep_routed_bias",
          {
              "model_name": "deepseek3-test",
              "dtype": "float32",
              "scan_layers": False,
              "first_num_dense_layers": 0,
              "custom_mesh_and_rule": "cp-as-ep",
              "ici_context_parallelism": 2,
              "ici_expert_parallelism": 2,
              "routed_bias": True,
              "routed_bias_update_rate": 0.001,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": True},
          1e-5,
          1e-5,
      ),
      (
          "row_major_only_single_chunk",
          {
              "dtype": "float32",
              "scan_layers": False,
              "ici_expert_parallelism": 4,
              "num_moe_token_chunks": 1,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "none", "ring_of_experts_row_major_reduce_scatter": True},
          1e-5,
          1e-5,
      ),
      (
          "software_pipeline_row_major_combine_bwd_rowwise",
          {
              "dtype": "bfloat16",
              "float32_gate_logits": True,
              "ici_expert_parallelism": 4,
              "moe_quantize_combine_bwd_method": "rowwise",
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
          },
          {"moe_ring_pipeline_mode": "software_pipeline", "ring_of_experts_row_major_reduce_scatter": True},
          5e-3,
          2e-3,
      ),
  )
  def test_modes_and_topologies_forward_backward_parity_and_hlo(self, common_overrides, test_overrides, rtol, atol):
    ref_diag, new_diag = self._assert_parity_against_baseline(4, common_overrides, test_overrides, rtol=rtol, atol=atol)
    n_chunks = common_overrides.get("num_moe_token_chunks", 2)
    mode = test_overrides["moe_ring_pipeline_mode"]
    row_major = test_overrides["ring_of_experts_row_major_reduce_scatter"]

    # Baseline ("none", row_major=False) has zero barriers, zero combine_order tags, and zero layout constraints.
    self.assertEqual(ref_diag["fwd_hlo_opt_barriers"], 0)
    self.assertEqual(ref_diag["vjp_pullback_opt_barriers"], 0)
    self.assertEqual(ref_diag["combine_order_names"], 0)
    self.assertEqual(ref_diag["layout_constraints"], 0)

    # Critical HLO & JAXpr structure invariants on the tested mode.
    expected_fwd_barriers = 4 * (n_chunks - 1) if (mode == "software_pipeline" and n_chunks > 1) else 0
    self.assertEqual(new_diag["fwd_jaxpr_barriers"], expected_fwd_barriers)
    self.assertEqual(new_diag["fwd_hlo_opt_barriers"], expected_fwd_barriers)
    # CRITICAL INVARIANT: ZERO backward opt-barriers in VJP pullback and jax.grad!
    self.assertEqual(new_diag["vjp_pullback_opt_barriers"], 0)
    self.assertEqual(new_diag["grad_hlo_total_opt_barriers"], expected_fwd_barriers)
    self.assertEqual(new_diag["unread_barriers"], [])

    expected_combine_order = (n_chunks - 1) if (mode == "combine_per_chunk" and n_chunks > 1) else 0
    self.assertEqual(new_diag["combine_order_names"], expected_combine_order)

    expected_layouts = n_chunks if row_major else 0
    self.assertEqual(new_diag["layout_constraints"], expected_layouts)
    self.assertEqual(new_diag["not_row_major"], [])

  def test_forced_routing_and_remat_tags_parity(self):
    common = {
        "dtype": "float32",
        "scan_layers": False,
        "ici_expert_parallelism": 4,
        "remat_policy": "custom",
        "moe_routing_maps": "device",
        "moe_x_sorted": "device",
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
    }
    cfg_ref = _moe_cfg("forced_ref", num_devices=4, **common)
    cfg_pipe = _moe_cfg(
        "forced_pipe",
        num_devices=4,
        **common,
        moe_ring_pipeline_mode="software_pipeline",
        ring_of_experts_row_major_reduce_scatter=True,
    )
    cfg_comb = _moe_cfg(
        "forced_comb",
        num_devices=4,
        **common,
        moe_ring_pipeline_mode="combine_per_chunk",
        ring_of_experts_row_major_reduce_scatter=True,
    )
    mesh = _make_mesh(cfg_ref, 4)
    x, upstream, forced = _make_inputs(cfg_ref, 4, forced_mode=True)
    ref_vals, _, shared_params = _run_layer_and_grads(cfg_ref, mesh, x, upstream, forced=forced)
    for cfg_test in (cfg_pipe, cfg_comb):
      new_vals, new_diag, _ = _run_layer_and_grads(
          cfg_test, mesh, x, upstream, forced=forced, shared_params=shared_params
      )
      np.testing.assert_allclose(
          np.asarray(new_vals[1], np.float32), np.asarray(ref_vals[1], np.float32), rtol=1e-5, atol=1e-5
      )
      np.testing.assert_allclose(
          np.asarray(new_vals[5], np.float32), np.asarray(ref_vals[5], np.float32), rtol=1e-5, atol=1e-5
      )
      self.assertEqual(new_diag["vjp_pullback_opt_barriers"], 0)
      self.assertEqual(new_diag["unread_barriers"], [])

  def test_packed_routing_gather_with_ring_pipeline_modes(self):
    """Verifies `moe_roe_packed_routing_gather=True` composes cleanly with both Ring-of-Experts pipeline modes."""
    common = {
        "dtype": "float32",
        "scan_layers": False,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "num_moe_token_chunks": 4,
        "moe_topk_before_ep_all_gather": True,
        "moe_roe_packed_routing_gather": True,
        "remat_policy": "custom",
        "moe_routing_maps": "device",
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
    }
    for mode in ("software_pipeline", "combine_per_chunk"):
      ref_diag, new_diag = self._assert_parity_against_baseline(
          4,
          common,
          {"moe_ring_pipeline_mode": mode, "ring_of_experts_row_major_reduce_scatter": True},
          rtol=1e-5,
          atol=1e-5,
      )
      self.assertEqual(ref_diag["fwd_hlo_opt_barriers"], 0)
      self.assertEqual(new_diag["vjp_pullback_opt_barriers"], 0)
      self.assertEqual(new_diag["unread_barriers"], [])
      self.assertEqual(new_diag["layout_constraints"], 4)
      if mode == "software_pipeline":
        self.assertEqual(new_diag["fwd_hlo_opt_barriers"], 12)
        self.assertEqual(new_diag["grad_hlo_total_opt_barriers"], 12)
      else:
        self.assertEqual(new_diag["combine_order_names"], 3)

  @parameterized.named_parameters(
      ("software_pipeline_full_remat", "software_pipeline", "full"),
      ("combine_per_chunk_full_remat", "combine_per_chunk", "full"),
      ("combine_per_chunk_minimal_remat", "combine_per_chunk", "minimal"),
      ("combine_per_chunk_custom_remat", "combine_per_chunk", "custom"),
      ("combine_per_chunk_minimal_offloaded_remat", "combine_per_chunk", "minimal_offloaded"),
      ("combine_per_chunk_qkv_proj_offloaded_remat", "combine_per_chunk", "qkv_proj_offloaded"),
  )
  def test_rematerialized_backward_emits_no_extra_reduce_scatters(self, mode, remat_policy):
    """Under Decoder.get_remat_policy() and scan-safe policy, neither mode replays chunk c-1's combine RS in backward."""
    base_kw = {
        "ici_expert_parallelism": 4,
        "num_moe_token_chunks": 2,
        "remat_policy": remat_policy,
        "ring_of_experts_row_major_reduce_scatter": True,
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
    }
    cfg_plain = _moe_cfg("remat_plain", num_devices=4, **{**base_kw, "ring_of_experts_row_major_reduce_scatter": False})
    cfg_mode = _moe_cfg(f"remat_{mode}_{remat_policy}", num_devices=4, **base_kw, moe_ring_pipeline_mode=mode)
    mesh = _make_mesh(cfg_plain, 4)
    x, upstream, _ = _make_inputs(cfg_plain, 4)

    if remat_policy in ("minimal_offloaded", "qkv_proj_offloaded") and mode == "combine_per_chunk":
      s_plain, o_plain = maxtext_utils.get_save_and_offload_names(cfg_plain)
      s_mode, o_mode = maxtext_utils.get_save_and_offload_names(cfg_mode)
      self.assertEqual(s_plain, [])
      self.assertEqual(s_mode, ["moe_combine_order"])
      self.assertEqual(o_mode, o_plain)

    is_host_offload = remat_policy in ("minimal_offloaded", "qkv_proj_offloaded")

    def _compile_remat_grad(cfg, policy_override="use_decoder", scan_safe=False):
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        decoder = nnx_decoders.NNXDecoder(cfg, mesh, rngs=nnx.Rngs(params=0))
        if policy_override != "use_decoder":
          policy = policy_override
        elif scan_safe:
          policy = decoder._get_scan_safe_remat_policy()  # pylint: disable=protected-access
        else:
          policy = decoder.get_remat_policy()
        graphdef, params, rest = nnx.split(_build_moe(cfg, mesh), nnx.Param, ...)

        @functools.partial(jax.checkpoint, policy=policy)
        def layer_fn(p, act):
          out, lb_loss, _ = nnx.merge(graphdef, p, rest)(act)
          return out, lb_loss

        def loss_fn(p, act):
          out, lb_loss = layer_fn(p, act)
          loss = jnp.sum(out.astype(jnp.float32) * upstream)
          return loss if lb_loss is None else loss + lb_loss.astype(jnp.float32)

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
        lowered = jax.jit(grad_fn).lower(params, x)
        # CPU backend has no pinned_host memory space for .compile() when host-offload policies are active.
        text = lowered.as_text("hlo") if (is_host_offload and not scan_safe) else (lowered.compile().as_text() or "")
        return _collectives(text)

    for scan_safe in (False, True):
      plain_collectives = _compile_remat_grad(cfg_plain, scan_safe=scan_safe)
      mode_collectives = _compile_remat_grad(cfg_mode, scan_safe=scan_safe)
      self.assertEqual(mode_collectives, plain_collectives)

    if mode == "combine_per_chunk" and not is_host_offload:
      # Positive control: dropping moe_combine_order from remat policy forces backward to replay combine RS!
      unsaved_collectives = _compile_remat_grad(cfg_mode, policy_override=jax.checkpoint_policies.nothing_saveable)
      self.assertGreater(
          sum(v for (op, _), v in unsaved_collectives.items() if op == "reduce-scatter"),
          sum(v for (op, _), v in plain_collectives.items() if op == "reduce-scatter"),
      )

  def test_scanned_qwen3_5_block_offloaded_and_full_remat_no_extra_reduce_scatters(self):
    """Scanned Qwen3.5 block preserves `moe_combine_order` across local-layer offload remat and global scan-safe remat."""
    q_kw = {
        "base_num_decoder_layers": 4,
        "inhomogeneous_layer_cycle_interval": 4,
        "scan_layers": True,
        "param_scan_axis": 0,
        "use_mrope": False,
        "attention": "dot_product",
        "base_num_query_heads": 4,
        "base_num_kv_heads": 2,
        "head_dim": 32,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 4,
        "gdn_key_head_dim": 32,
        "gdn_value_head_dim": 32,
        "gdn_chunk_size": 16,
        "ici_expert_parallelism": 4,
        "num_moe_token_chunks": 2,
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
    }

    def _qwen_rs_count(cfg, override=None):
      mesh = _make_mesh(cfg, 4)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        dec = nnx_decoders.NNXDecoder(cfg, mesh, model_mode=ctypes.MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
        blk = qwen3_5.Qwen3_5ScannableBlock(
            config=cfg,
            mesh=mesh,
            model_mode=ctypes.MODEL_MODE_TRAIN,
            quant=None,
            rngs=nnx.Rngs(params=0, dropout=1),
            remat_policy_fn=dec.get_remat_policy(),
            apply_internal_remat=True,
        )
        if override is not None:
          dec.remat_policy_override = override
          dec._sync_remat_policy_override(blk)  # pylint: disable=protected-access
        graphdef, params, rest = nnx.split(blk, nnx.Param, ...)
        y = jnp.ones((4, _SEQ, _EMB), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32), (4, _SEQ))
        seg = jnp.ones((4, _SEQ), dtype=jnp.int32)

        def nnx_loss(b, act_in):
          out = b(act_in, seg, pos, True, ctypes.MODEL_MODE_TRAIN)
          if isinstance(out, tuple):
            out = out[0]
          return jnp.sum(out.astype(jnp.float32) ** 2)

        grad_fn = nnx.value_and_grad(nnx_loss, argnums=nnx.DiffState(0, nnx.Param))

        def jitted(p, r, act):
          return grad_fn(nnx.merge(graphdef, p, r), act)

        hlo = jax.jit(jitted).lower(params, rest, y).as_text("hlo")
        return sum(v for (op, _), v in _collectives(hlo).items() if op == "reduce-scatter")

    for pol_name in ("minimal_offloaded", "qkv_proj_offloaded"):
      cfg_p = _moe_cfg(f"qwen_p_{pol_name}", num_devices=4, **{**q_kw, "remat_policy": pol_name})
      cfg_c = _moe_cfg(
          f"qwen_c_{pol_name}",
          num_devices=4,
          **{**q_kw, "remat_policy": pol_name},
          moe_ring_pipeline_mode="combine_per_chunk",
      )
      self.assertEqual(_qwen_rs_count(cfg_c), _qwen_rs_count(cfg_p))

    cfg_p_ov = _moe_cfg("qwen_p_ov_full", num_devices=4, **{**q_kw, "remat_policy": "minimal_offloaded"})
    cfg_c_ov = _moe_cfg(
        "qwen_c_ov_full",
        num_devices=4,
        **{**q_kw, "remat_policy": "minimal_offloaded"},
        moe_ring_pipeline_mode="combine_per_chunk",
    )
    self.assertEqual(_qwen_rs_count(cfg_c_ov, override="full"), _qwen_rs_count(cfg_p_ov, override="full"))

  def test_llama4_non_ragged_combine_ordering_preserved_in_hlo(self):
    """Llama4 with `use_ragged_sort=False` ties `intermediate_output` so combine ordering survives DCE."""
    base_common = {
        "model_name": "llama4-17b-16e",
        "num_experts_per_tok": 1,
        "dtype": "float32",
        "scan_layers": False,
        "base_num_decoder_layers": 4,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 2,
        "ici_expert_parallelism": 4,
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
    }

    def _compiled_rs_user_counts(cfg, mesh, params, act, upstream):
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        gdef, _, rest = nnx.split(_build_moe(cfg, mesh), nnx.Param, ...)

        def loss_fn(p, inp):
          out, _, _ = nnx.merge(gdef, p, rest)(inp)
          return jnp.mean(out.astype(jnp.float32) * upstream)

        hlo = jax.jit(loss_fn).lower(params, act).compile().as_text() or ""
      lines = [line.strip() for line in hlo.splitlines()]
      rs_vars = [line.split(" = ", 1)[0] for line in lines if " reduce-scatter(" in line]
      counts = []
      for rs_var in rs_vars:
        pattern = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(rs_var)}(?![A-Za-z0-9_.])")
        users = [line for line in lines if not line.startswith(f"{rs_var} =") and pattern.search(line)]
        counts.append(len(users))
      return counts

    for n_chunks in (2, 4):
      common = {**base_common, "num_moe_token_chunks": n_chunks}
      for mode in ("combine_per_chunk", "software_pipeline"):
        _, new_diag = self._assert_parity_against_baseline(
            4,
            common,
            {"moe_ring_pipeline_mode": mode, "ring_of_experts_row_major_reduce_scatter": True},
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(new_diag["vjp_pullback_opt_barriers"], 0)
        self.assertEqual(new_diag["unread_barriers"], [])
        if mode == "software_pipeline":
          self.assertEqual(new_diag["fwd_hlo_opt_barriers"], 4 * (n_chunks - 1))
          hlo_lines = [line.strip() for line in new_diag["fwd_hlo"].splitlines()]
          bar_vars = [line.split(" = ", 1)[0] for line in hlo_lines if "opt-barrier(" in line]
          self.assertEqual(len(bar_vars), 4 * (n_chunks - 1))
          for bar_var in bar_vars:
            bar_pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(bar_var)}(?![A-Za-z0-9_.])")
            gte_lines = [l for l in hlo_lines if not l.startswith(f"{bar_var} =") and bar_pat.search(l)]
            self.assertEqual(len(gte_lines), 2, msg=f"Expected 2 tuple-element users for {bar_var}: {gte_lines}")
            for gte_line in gte_lines:
              gte_var = gte_line.split(" = ", 1)[0]
              gte_pat = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(gte_var)}(?![A-Za-z0-9_.])")
              downstream = [l for l in hlo_lines if not l.startswith(f"{gte_var} =") and gte_pat.search(l)]
              self.assertGreaterEqual(len(downstream), 1, msg=f"Dead barrier output {gte_line}")
        else:
          self.assertEqual(new_diag["combine_order_names"], n_chunks - 1)
          cfg_ref = _moe_cfg(f"l4_ref_{n_chunks}", num_devices=4, **common)
          cfg_comb = _moe_cfg(
              f"l4_comb_{n_chunks}",
              num_devices=4,
              **common,
              moe_ring_pipeline_mode="combine_per_chunk",
          )
          mesh = _make_mesh(cfg_ref, 4)
          x, upstream, _ = _make_inputs(cfg_ref, 4)
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
            _, shared_params, _ = nnx.split(_build_moe(cfg_ref, mesh), nnx.Param, ...)
          ref_rs_users = _compiled_rs_user_counts(cfg_ref, mesh, shared_params, x, upstream)
          comb_rs_users = _compiled_rs_user_counts(cfg_comb, mesh, shared_params, x, upstream)
          self.assertEqual(ref_rs_users, [1] * n_chunks)
          self.assertEqual(sum(c >= 2 for c in comb_rs_users), n_chunks - 1, msg=f"comb_rs_users={comb_rs_users}")

    # Verify full remat on Llama4 combine_per_chunk saves only 0-D scalar ordering tokens and emits no extra RS.
    remat_kw = {**base_common, "num_moe_token_chunks": 2, "remat_policy": "full"}
    cfg_r_ref = _moe_cfg("l4_remat_ref", num_devices=4, **remat_kw)
    cfg_r_comb = _moe_cfg("l4_remat_comb", num_devices=4, **remat_kw, moe_ring_pipeline_mode="combine_per_chunk")
    mesh = _make_mesh(cfg_r_ref, 4)
    x, upstream, _ = _make_inputs(cfg_r_ref, 4)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_r_ref.logical_axis_rules):
      _, shared_params, _ = nnx.split(_build_moe(cfg_r_ref, mesh), nnx.Param, ...)

      def _remat_info(cfg):
        dec = nnx_decoders.NNXDecoder(cfg, mesh, rngs=nnx.Rngs(params=0))
        gdef, _, rest = nnx.split(_build_moe(cfg, mesh), nnx.Param, ...)

        @functools.partial(jax.checkpoint, policy=dec.get_remat_policy())
        def ckpt_fn(p, inp):
          out, _, _ = nnx.merge(gdef, p, rest)(inp)
          return out

        def loss_fn(p, inp):
          return jnp.sum(ckpt_fn(p, inp).astype(jnp.float32) * upstream)

        _, vjp_fn = jax.vjp(ckpt_fn, shared_params, x)
        res_shapes = [tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(vjp_fn) if isinstance(leaf, jax.Array)]
        grad_hlo = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1))).lower(shared_params, x).compile().as_text() or ""
        return _collectives(grad_hlo), res_shapes

      ref_coll, ref_res = _remat_info(cfg_r_ref)
      comb_coll, comb_res = _remat_info(cfg_r_comb)
    self.assertEqual(comb_coll, ref_coll)
    extra_res = list(comb_res)
    for shp in ref_res:
      if shp in extra_res:
        extra_res.remove(shp)
    # No `(buffer, emb)`-sized residual is added; at most the 0-D `moe_combine_order` token.
    self.assertTrue(all(math.prod(shp) <= 1 for shp in extra_res), msg=f"extra residuals: {extra_res}")

  def test_barrier_removal_flag_detection_and_override_precedence(self):
    flag = "--xla_tpu_aggressive_opt_barrier_removal"
    cases = (
        ("", "", True),
        (f"{flag}=true", "", False),
        ("", f"{flag}=true", False),
        ("", f"{flag}=ENABLED", False),
        (f"{flag}=false", f"{flag}=true", True),
        (f"{flag}=true", f"{flag}=false", False),
    )
    for compile_flags, env_flags, should_pipeline in cases:
      cfg = _moe_cfg(
          "barrier_flag",
          num_devices=4,
          ici_expert_parallelism=4,
          moe_ring_pipeline_mode="software_pipeline",
          compile_xla_flags=compile_flags,
          use_ragged_sort=False,
          ragged_buffer_factor=-1.0,
      )
      mesh = _make_mesh(cfg, 4)
      x, upstream, _ = _make_inputs(cfg, 4)
      with (
          mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": env_flags, "XLA_FLAGS": ""}),
          mock.patch.object(moe.max_logging, "warning") as mock_warn,
      ):
        _, diag, _ = _run_layer_and_grads(cfg, mesh, x, upstream)
      expected = 4 * (cfg.num_moe_token_chunks - 1) if should_pipeline else 0
      self.assertEqual(diag["fwd_hlo_opt_barriers"], expected)
      warned = any("moe_ring_pipeline_mode='software_pipeline' is skipped" in str(c) for c in mock_warn.call_args_list)
      self.assertEqual(warned, not should_pipeline)

  def test_dynamic_qwag_amax_threaded_across_pipeline_modes(self):
    """Verifies `moe_qwag_weight_amax` (`rhs_amax`) is threaded into `_up_projection` and `_down_projection`."""
    common = {
        "dtype": "float32",
        "scan_layers": False,
        "ici_fsdp_parallelism": 2,
        "ici_expert_parallelism": 2,
        "num_moe_token_chunks": 2,
        "use_ragged_sort": False,
        "ragged_buffer_factor": -1.0,
        "megablox": True,
        "use_tokamax_gmm": False,
        "use_gmm_v2": False,
    }
    seen_amax_shapes: list[tuple[int, ...] | None] = []
    real_gmm = moe.mblx.gmm

    def spy_gmm(lhs, rhs, group_sizes, *args, **kwargs):
      rhs_amax = kwargs.get("rhs_amax")
      seen_amax_shapes.append(None if rhs_amax is None else tuple(rhs_amax.shape))
      # Touch rhs_amax inside shard_map so any outer-tracer leak or VMA mismatch fails immediately.
      if rhs_amax is not None:
        lhs = lhs + 0.0 * jnp.sum(rhs_amax).astype(lhs.dtype)
      kwargs_clean = dict(kwargs)
      kwargs_clean["rhs_amax"] = None
      kwargs_clean["rhs_scale_kept_axes"] = ()
      return real_gmm(lhs, rhs, group_sizes, *args, **kwargs_clean)

    cfg_ref = _moe_cfg("qwag_ref", num_devices=4, **common)
    mesh = _make_mesh(cfg_ref, 4)
    x, upstream, _ = _make_inputs(cfg_ref, 4)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      ref_model = _build_moe(cfg_ref, mesh)
      _, shared_params, _ = nnx.split(ref_model, nnx.Param, ...)
      local_e = _NUM_EXPERTS // 2
      # Per-channel QWAG amax tuple `(wi_0_amax, wi_1_amax, wo_amax)` matching `[E, 1, N]` / `[1, 1, N]`.
      qwag_amax = (
          jnp.ones((1, 1, _MLP), dtype=jnp.float32),
          jnp.ones((1, 1, _MLP), dtype=jnp.float32) * 2.0,
          jnp.ones((1, 1, _EMB), dtype=jnp.float32) * 3.0,
      )
      assert local_e > 0

      def run_mode(mode: str):
        seen_amax_shapes.clear()
        cfg_m = _moe_cfg(
            f"qwag_{mode}",
            num_devices=4,
            **common,
            moe_ring_pipeline_mode=mode,
            ring_of_experts_row_major_reduce_scatter=(mode != "none"),
        )
        m_inst = _build_moe(cfg_m, mesh)
        gdef, _, rest = nnx.split(m_inst, nnx.Param, ...)

        def loss_fn(p, act, amax_tuple):
          mod = nnx.merge(gdef, p, rest)
          # Exercise sparse_matmul directly with explicit precomputed `moe_qwag_weight_amax` by patching
          # `explicitly_weight_ag` via a dynamic QWAG rule context on `qpl.get_current_rule`.
          out, _, _ = mod(act, moe_qwag_weight_amax=amax_tuple)
          return jnp.mean(out.astype(jnp.float32) * upstream), out

        fake_rule = types.SimpleNamespace(weight_calibration_method="absmax")
        with (
            mock.patch.object(moe.ctypes, "moe_qwag_flag", return_value="shard_embed_moe_on_fsdp"),
            mock.patch.object(moe.qpl, "get_current_rule", return_value=fake_rule),
            mock.patch.object(moe.mblx, "gmm", side_effect=spy_gmm),
        ):
          (loss_v, out_v), (pgrads_v, xgrad_v) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(
              shared_params, x, qwag_amax
          )
        return (loss_v, out_v, pgrads_v, xgrad_v), list(seen_amax_shapes)

      ref_out, ref_shapes = run_mode("none")
      comb_out, comb_shapes = run_mode("combine_per_chunk")
      pipe_out, pipe_shapes = run_mode("software_pipeline")

    # 2 chunks * 3 GMMs (`wi_0`, `wi_1`, `wo`) = 6 forward GMM traces, every one receiving non-None `rhs_amax`.
    self.assertEqual(len(ref_shapes), 6)
    self.assertTrue(all(s is not None for s in ref_shapes), msg=f"Missing rhs_amax in 'none': {ref_shapes}")
    self.assertEqual(comb_shapes, ref_shapes)
    self.assertEqual(pipe_shapes, ref_shapes)
    for got_out in (comb_out, pipe_out):
      np.testing.assert_allclose(np.asarray(got_out[1]), np.asarray(ref_out[1]), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.asarray(got_out[3]), np.asarray(ref_out[3]), rtol=1e-5, atol=1e-5)

  @pytest.mark.cpu_only
  def test_8_device_cpu_mesh_bfloat16_bit_exact(self):
    """Runs an 8-device CPU child process with `--xla_allow_excess_precision=false` for bit-identical bf16 checks."""
    # Test runners' XML / sharding outputs belong to the parent, so the child must not write them.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("XML_OUTPUT_FILE", "TEST_SHARD_INDEX", "TEST_TOTAL_SHARDS", "TEST_SHARD_STATUS_FILE")
    }
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8 --xla_allow_excess_precision=false"
    )
    env["JAX_PLATFORMS"] = "cpu"
    env[_SUBPROC_ENV] = "1"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    py_paths = [repo_root] + [p for p in sys.path if p]
    if env.get("PYTHONPATH"):
      py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(py_paths))
    # Hermetic test binaries may have no `sys.executable`; re-run the binary itself in that case.
    cmd = [sys.executable, os.path.abspath(__file__)] if sys.executable else [sys.argv[0]]
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    self.assertIn(_SUBPROC_PASS_TOKEN, result.stdout)


def _run_8dev_bit_exact_main():
  """Child-process entry point verifying bit-identical bf16 forward outputs across 8 CPU devices (`ep=8` & `fsdp=2,ep=4`)."""
  assert len(jax.devices()) == 8, f"Expected 8 CPU devices, got {len(jax.devices())}"
  topologies = (
      ("ep8", {"ici_fsdp_parallelism": 1, "ici_expert_parallelism": 8, "per_device_batch_size": 1.0}),
      ("fsdp2_ep4", {"ici_fsdp_parallelism": 2, "ici_expert_parallelism": 4, "per_device_batch_size": 1.0}),
  )
  modes = (
      ("combine_per_chunk", False),
      ("combine_per_chunk", True),
      ("software_pipeline", False),
      ("software_pipeline", True),
      ("none", True),
  )
  for topo_name, topo_kw in topologies:
    common = {"dtype": "bfloat16", "use_random_routing": True, **topo_kw}
    cfg_ref = _moe_cfg(f"sub_ref_{topo_name}", num_devices=8, **common)
    mesh = _make_mesh(cfg_ref, 8)
    x, upstream, _ = _make_inputs(cfg_ref, 8)
    ref_vals, _, shared_params = _run_layer_and_grads(cfg_ref, mesh, x, upstream)
    for mode, row_major in modes:
      cfg_new = _moe_cfg(
          f"sub_{topo_name}_{mode}_{row_major}",
          num_devices=8,
          **common,
          moe_ring_pipeline_mode=mode,
          ring_of_experts_row_major_reduce_scatter=row_major,
      )
      new_vals, diag, _ = _run_layer_and_grads(cfg_new, mesh, x, upstream, shared_params=shared_params)
      assert np.array_equal(
          np.asarray(new_vals[1]), np.asarray(ref_vals[1])
      ), f"{topo_name} {mode} {row_major} output mismatch"
      assert diag["vjp_pullback_opt_barriers"] == 0, f"{topo_name} {mode} backward barriers != 0"
      assert not diag["unread_barriers"], f"{topo_name} {mode} unread barriers: {diag['unread_barriers']}"
      for g_ref, g_new in zip(
          jax.tree_util.tree_leaves(ref_vals[4]), jax.tree_util.tree_leaves(new_vals[4]), strict=True
      ):
        np.testing.assert_allclose(np.asarray(g_new, np.float32), np.asarray(g_ref, np.float32), rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose(
          np.asarray(new_vals[5], np.float32), np.asarray(ref_vals[5], np.float32), rtol=1e-5, atol=1e-6
      )
  print(_SUBPROC_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  if os.environ.get(_SUBPROC_ENV) == "1":
    app.run(lambda _: _run_8dev_bit_exact_main())
  else:
    absltest.main()

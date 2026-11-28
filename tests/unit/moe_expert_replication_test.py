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

"""CPU and TPU tests for RoutedMoE online hot-expert replication wiring."""

import contextlib
import functools
import math
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import qwix
import qwix.pallas as qpl

from maxtext.configs import pyconfig
from maxtext.kernels.megablox import ops as mblx_ops
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import moe_replica_planner as mrp
from maxtext.layers import moe_replica_weights as mrw
from maxtext.layers import quantizations
from tests.utils import moe_replica_test_utils
from tests.utils.test_helpers import get_test_config_path


def _pure_jax_gmm(lhs, rhs, group_sizes, preferred_element_type=jnp.float32, existing_out=None, **_unused_kwargs):
  """Differentiable group-wise matmul matching megablox.gmm semantics on CPU."""
  starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes.astype(jnp.int32))])
  rows = jnp.arange(lhs.shape[0], dtype=jnp.int32)
  out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=preferred_element_type) if existing_out is None else existing_out
  for g in range(rhs.shape[0]):
    in_group = (rows >= starts[g]) & (rows < starts[g + 1])
    proj = jnp.dot(lhs, rhs[g], precision=jax.lax.Precision.HIGHEST).astype(preferred_element_type)
    out = jnp.where(in_group[:, None], proj, out)
  return out


def _spec_dim_axes(spec, mesh_sizes):
  """Returns per-dimension active sharding axes `((ax, ...), ...)` for `spec`."""
  if not spec:
    return ()
  parts = spec.partitions if isinstance(spec, P) else tuple(spec)
  out = []
  for entry in parts:
    items = entry if isinstance(entry, tuple) else ((entry,) if entry else ())
    out.append(tuple(ax for ax in items if mesh_sizes.get(ax, 1) > 1))
  return tuple(out)


def _spec_axes(spec, mesh_sizes, mesh_axes):
  """Returns active sharding axis names anywhere in `spec`, ordered by `mesh_axes`."""
  present = set()
  for dim_axes in _spec_dim_axes(spec, mesh_sizes):
    present.update(dim_axes)
  return tuple(ax for ax in mesh_axes if ax in present)


def _init_moe_cfg(run_name, **overrides):
  """Initializes a small Mixtral sparse-matmul pyconfig with `overrides`."""
  kwargs = {
      "run_name": run_name,
      "base_output_directory": f"/tmp/{run_name}",
      "enable_checkpointing": False,
      "model_name": "mixtral-8x7b",
      "override_model_config": True,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      "megablox": True,
      "sparse_matmul": True,
      "use_ring_of_experts": False,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "load_balance_loss_weight": 0.0,
      "routed_bias": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([None, get_test_config_path()], **kwargs)


def _call_moe_body_return_weights(fn, s_args):
  """Calls inner _moe_body with return_weights=True to exercise the chunk-wgrad guardrail.

  _moe_body is nested inside sparse_matmul and only receives return_weights=True under
  use_ring_of_experts=True (which fails earlier in route()), so closure lookup reaches it directly.
  """
  moe_body = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))["_moe_body"]
  return moe_body(*s_args[:12], return_weights=True, per_expert_scale=s_args[12], token_mask=s_args[13])[:6]


class RoutedMoEReplicationWiringTest(parameterized.TestCase):
  """CPU unit tests exercising RoutedMoE hot-expert replication paths directly."""

  def _make_layer(self, *, num_ep=2, num_experts=4, num_experts_per_tok=2, emb_dim=16, mlp_dim=8, **overrides):
    """Builds a float32 RoutedMoE layer configured for multi-shard EP emulation on CPU."""
    cfg_kw = {
        "base_emb_dim": emb_dim,
        "base_moe_mlp_dim": mlp_dim,
        "base_mlp_dim": mlp_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "per_device_batch_size": 2,
        "max_target_length": 4,
        "compile_topology": "v5e-8",
        "compile_topology_num_slices": 1,
        "ici_expert_parallelism": max(num_ep, 2),
        "ici_fsdp_parallelism": 8 // max(num_ep, 2),
        **overrides,
    }
    cfg = _init_moe_cfg("moe_replication_wiring_cpu", **cfg_kw)
    fake_mesh = types.SimpleNamespace(
        axis_names=("expert",),
        shape={"expert": num_ep},
        devices=np.array([jax.devices()[0]]),
        local_devices=[jax.devices()[0]],
        empty=False,
    )
    layer = moe.RoutedMoE(
        config=cfg,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        mesh=fake_mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=mlp_dim,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    layer._maybe_shard_with_pspec = lambda value, _pspec, **_kw: value  # pylint: disable=protected-access
    layer._maybe_shard_with_logical = lambda value, _axes: value  # pylint: disable=protected-access
    layer._logical_to_mesh_axes = lambda logical_axes: P(  # pylint: disable=protected-access
        *("expert" if axis in ("activation_batch", "exp", "exp_with_fsdp") else None for axis in logical_axes)
    )
    return layer

  @contextlib.contextmanager
  def _cpu_ep_runtime(
      self, layer, *, num_ep=2, cotangent=None, captured=None, wrap_shard_fn=None, gmm_impl=None, num_fsdp=1
  ):
    """Emulates multi-shard EP (and optional FSDP, including multi-dim QAG weight sharding) + SPMD VJP on CPU."""
    orig_hardware = layer.config.hardware
    orig_mesh = layer.mesh
    shard_gs_box = []
    mesh_axes = ("fsdp", "expert") if num_fsdp > 1 else ("expert",)
    mesh_sizes = {"fsdp": num_fsdp, "expert": num_ep}

    def _split(a, spec):
      kept = _spec_axes(spec, mesh_sizes, mesh_axes) if a is not None else ()
      if not kept:
        return a
      dim_axes = _spec_dim_axes(spec, mesh_sizes)
      shape, axis_pos, tensor_pos = [], {}, []
      for d in range(a.ndim):
        axs = dim_axes[d] if d < len(dim_axes) else ()
        shard_prod = math.prod(mesh_sizes[ax] for ax in axs)
        for ax in axs:
          axis_pos[ax] = len(shape)
          shape.append(mesh_sizes[ax])
        tensor_pos.append(len(shape))
        shape.append(a.shape[d] // shard_prod)
      return jnp.transpose(jnp.reshape(a, shape), [axis_pos[ax] for ax in kept] + tensor_pos)

    def _unsplit(val, spec, *, sum_replicated):
      if val is None:
        return None
      kept = _spec_axes(spec, mesh_sizes, mesh_axes)
      unkept = tuple(i for i, ax in enumerate(mesh_axes) if ax not in kept)
      if sum_replicated and unkept:
        val = jnp.sum(val, axis=unkept)
      elif not sum_replicated:
        for i in reversed(unkept):
          val = jnp.take(val, 0, axis=i)
      if not kept:
        return val
      dim_axes = _spec_dim_axes(spec, mesh_sizes)
      kept_pos = {ax: i for i, ax in enumerate(kept)}
      base = len(kept)
      perm, merged = [], []
      for d in range(val.ndim - base):
        axs = dim_axes[d] if d < len(dim_axes) else ()
        perm.extend(kept_pos[ax] for ax in axs)
        perm.append(base + d)
        merged.append(val.shape[base + d] * math.prod(mesh_sizes[ax] for ax in axs))
      return jnp.reshape(jnp.transpose(val, perm), merged)

    def _vmap_axes(fn, specs):
      for ax in reversed(mesh_axes):
        fn = jax.vmap(
            fn,
            axis_name=ax,
            in_axes=tuple(0 if ax in _spec_axes(s, mesh_sizes, mesh_axes) else None for s in specs),
            axis_size=mesh_sizes[ax],
        )
      return fn

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
      del mesh, check_vma

      def run_shard(s_args):
        shard_gs_box.clear()
        outs = wrap_shard_fn(fn, s_args) if wrap_shard_fn is not None else fn(*s_args)
        return outs, (shard_gs_box[0] if shard_gs_box else None)

      def wrapped(*args):
        split_args = [_split(a, s) for a, s in zip(args, in_specs)]
        if cotangent is not None:
          cot_shards = _split(cotangent, out_specs[0])

          def per_shard_fwd_bwd(*all_s):
            *s_args, cot_s = all_s

            def diff_fn(x_, g_, w0_, w1_, wo_):
              outs, gs = run_shard((x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:]))
              return outs[0], (outs, gs)

            _, vjp_fn, (outs, gs) = jax.vjp(diff_fn, s_args[0], s_args[1], s_args[3], s_args[4], s_args[5], has_aux=True)
            return outs, gs, vjp_fn(cot_s)

          raw_outs, local_gs, shard_grads = _vmap_axes(per_shard_fwd_bwd, [*in_specs, out_specs[0]])(
              *split_args, cot_shards
          )
          if captured is not None:
            grad_specs = (in_specs[0], in_specs[1], in_specs[3], in_specs[4], in_specs[5])
            captured["grads"] = tuple(_unsplit(g, s, sum_replicated=True) for g, s in zip(shard_grads, grad_specs))
        else:
          raw_outs, local_gs = _vmap_axes(lambda *s_args: run_shard(s_args), in_specs)(*split_args)
        if captured is not None:
          captured["local_group_sizes"] = local_gs
        return tuple(_unsplit(v, s, sum_replicated=False) for v, s in zip(raw_outs, out_specs))

      return wrapped

    def tracked_gmm(*args, **kwargs):
      if not shard_gs_box and wrap_shard_fn is None:
        gs = kwargs.get("group_sizes", args[2] if len(args) > 2 else None)
        if gs is not None:
          shard_gs_box.append(gs)
      return (gmm_impl or _pure_jax_gmm)(*args, **kwargs)

    try:
      object.__setattr__(layer.config, "hardware", "tpu")
      if num_fsdp > 1:
        layer.mesh = types.SimpleNamespace(
            axis_names=("fsdp", "expert"),
            shape={"fsdp": num_fsdp, "expert": num_ep},
            devices=orig_mesh.devices,
            local_devices=orig_mesh.local_devices,
            empty=False,
        )
      with (
          mock.patch.object(jax, "shard_map", side_effect=fake_shard_map),
          mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=moe_replica_test_utils.emulated_ragged_all_to_all),
          mock.patch.object(moe.mblx, "gmm", side_effect=tracked_gmm),
      ):
        yield
    finally:
      object.__setattr__(layer.config, "hardware", orig_hardware)
      layer.mesh = orig_mesh

  def _sample_tensors(self, seed, num_experts=4, batch=4, expert0_mult=1.0):
    k_in, k_log, k_w0, k_w1, k_wo, k_cot = jax.random.split(jax.random.PRNGKey(seed), 6)
    mag = jnp.ones((num_experts, 1, 1), jnp.float32).at[0].set(expert0_mult)
    return (
        jax.random.normal(k_in, (batch, 4, 16), jnp.float32),
        jax.random.normal(k_log, (batch, 4, num_experts), jnp.float32),
        jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.25 * mag,
        jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.25 * mag,
        jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.25 * mag,
        jax.random.normal(k_cot, (batch, 4, 16), jnp.float32),
    )

  def test_flag_off_skips_replication_modules(self):
    """With expert_replicas_per_device=0, neither planner nor weight replication runs."""
    inputs, logits, w0, w1, wo, cotangent = self._sample_tensors(seed=3)
    layer_a = self._make_layer(expert_replicas_per_device=0)
    layer_b = self._make_layer(expert_replicas_per_device=0)
    with self._cpu_ep_runtime(layer_a, cotangent=cotangent):
      out_a, _, _ = layer_a.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)
    with (
        mock.patch.object(mrp, "compute_plan_iter", side_effect=AssertionError("planner must not run at R=0")),
        mock.patch.object(mrw, "extend_with_replicas", side_effect=AssertionError("weights must not run at R=0")),
        self._cpu_ep_runtime(layer_b, cotangent=cotangent),
    ):
      out_b, _, _ = layer_b.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)
    np.testing.assert_array_equal(np.asarray(out_b), np.asarray(out_a))

  def test_permute_with_physical_route_fn(self):
    """Verifies permute() remaps logical experts into EP*(L+R) physical bins via physical_route_fn."""
    layer = self._make_layer(num_ep=2, num_experts=4, num_experts_per_tok=2)
    inputs = jnp.arange(2 * 3 * 16, dtype=jnp.float32).reshape(2, 3, 16)
    gate_logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4), dtype=jnp.float32)
    forced = jnp.array([[[0, 1], [0, 2], [0, 3]], [[0, 1], [0, 2], [0, 3]]], dtype=jnp.int32)
    base_map = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
    sorted_inputs, _, _, group_sizes, *_ = layer.permute(
        inputs,
        gate_logits,
        None,
        use_custom_sort_vjp=False,
        rngs=jax.random.PRNGKey(1),
        forced_routed_experts=forced,
        compute_lb_loss=False,
        physical_route_fn=lambda flat, _mask=None: (
            jnp.where((flat == 0) & (jnp.arange(flat.shape[0]) >= 6), 5, base_map[flat]),
            6,
        ),
    )
    self.assertEqual(group_sizes.shape, (6,))
    self.assertEqual(int(jnp.sum(group_sizes)), forced.size)
    np.testing.assert_array_equal(np.asarray(group_sizes), np.array([3, 2, 0, 2, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(sorted_inputs[-3:]), np.asarray(inputs[1]), atol=1e-6)

  @parameterized.named_parameters(
      ("within_2048_uses_packed_sort", 2048, 2047),
      ("above_2048_falls_back_without_uint32_wrap", 4097, 4096),
  )
  def test_permute_packed_routing_sort_num_sort_bins_guard(self, total_bins, target_bin):
    """Verifies moe_packed_routing_sort checks num_sort_bins <= 2^11 (avoiding uint32 overflow at bin 4096)."""
    inputs = jnp.arange(2 * 2 * 16, dtype=jnp.float32).reshape(2, 2, 16)
    logits = jnp.zeros((2, 2, 4), dtype=jnp.float32).at[:, :, 0].set(10.0).at[:, :, 1].set(5.0)

    def route_fn(flat, _mask=None):
      return jnp.where(flat == 0, target_bin, flat), total_bins

    layer_ref = self._make_layer(moe_packed_routing_sort=False)
    layer_packed = self._make_layer(moe_packed_routing_sort=True)
    s_ref, _, _, gs_ref, *_ = layer_ref.permute(
        inputs, logits, None, False, jax.random.PRNGKey(0), compute_lb_loss=False, physical_route_fn=route_fn
    )
    s_pk, _, _, gs_pk, *_ = layer_packed.permute(
        inputs, logits, None, False, jax.random.PRNGKey(0), compute_lb_loss=False, physical_route_fn=route_fn
    )
    self.assertEqual(gs_pk.shape, (total_bins,))
    self.assertEqual(int(gs_pk[target_bin]), 4)
    np.testing.assert_array_equal(np.asarray(gs_pk), np.asarray(gs_ref))
    np.testing.assert_allclose(np.asarray(s_pk[-4:]), np.asarray(s_ref[-4:]), atol=1e-6)

  def test_end_to_end_sparse_matmul_parity_and_required_rbf(self):
    """Verifies forward output, (dx, dlogits, dw0, dw1, dwo), and lowered required_rbf match R=0 on skewed traffic."""
    inputs, logits, w0, w1, wo, cotangent = self._sample_tensors(seed=42)
    forced = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [0, 3]],
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )

    def run_arm(replicas):
      layer = self._make_layer(
          expert_replicas_per_device=replicas,
          expert_replica_max_send=2,
          log_required_ragged_buffer_factor=True,
      )
      cap = {}
      with self._cpu_ep_runtime(layer, cotangent=cotangent, captured=cap):
        out, _, _ = layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)
      req = float(np.asarray(nnx.pop(layer, nnx.Intermediate)["moe_required_rbf"].get_value()[0]))
      return out, cap, req

    out_ref, cap_ref, req_base = run_arm(0)
    out_rep, cap_rep, req_rep = run_arm(1)
    self.assertGreater(int(np.asarray(cap_rep["local_group_sizes"])[1, -1]), 0)
    self.assertLess(req_rep, req_base)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for name, g_rep, g_ref in zip(("dx", "dlogits", "dw0", "dw1", "dwo"), cap_rep["grads"], cap_ref["grads"]):
      with self.subTest(grad=name):
        np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("use_ring_of_experts", 2, {"use_ring_of_experts": True}, False, None, NotImplementedError, "ragged all-to-all"),
      ("num_moe_emb_chunks", 2, {"num_moe_emb_chunks": 2}, False, None, NotImplementedError, "num_moe_emb_chunks"),
      ("not_divisible", 2, {"num_experts": 5}, False, None, ValueError, "divisible by num_ep"),
      ("moe_bias", 2, {}, True, None, NotImplementedError, "MoE biases"),
      ("return_weights", 2, {}, False, _call_moe_body_return_weights, NotImplementedError, "moe_accumulate_chunk_wgrad"),
      ("ep_1_runtime", 1, {}, False, None, ValueError, "requires num_ep > 1"),
  )
  def test_runtime_guardrails_raise(self, num_ep, cfg_mut, pass_bias, shard_fn, exc_type, msg):
    inputs = jnp.ones((4, 4, 16), jnp.float32)
    n_exp = cfg_mut.get("num_experts", 4)
    logits = jnp.ones((4, 4, n_exp), jnp.float32)
    w0, w1, wo = jnp.ones((4, 16, 8)), jnp.ones((4, 16, 8)), jnp.ones((4, 8, 16))
    w0_bias = jnp.zeros((4, 8), jnp.float32) if pass_bias else None
    layer = self._make_layer(num_ep=num_ep, expert_replicas_per_device=1)
    for k, v in cfg_mut.items():
      object.__setattr__(layer.config, k, v)
    with self._cpu_ep_runtime(layer, num_ep=num_ep, wrap_shard_fn=shard_fn):
      with self.assertRaisesRegex(exc_type, msg):
        layer.sparse_matmul(inputs, logits, None, w0, w1, wo, w0_bias, None, None)

  @parameterized.named_parameters(
      ("w0_flat_2d", (64, 8), (4, 16, 8), (4, 8, 16), "3D local w0 weights with leading dim 2"),
      ("w1_wrong_leading_dim", (4, 16, 8), (8, 16, 8), (4, 8, 16), "3D local w1 weights with leading dim 2"),
      ("wo_wrong_leading_dim", (4, 16, 8), (4, 16, 8), (8, 8, 16), "3D local wo weights with leading dim 2"),
  )
  def test_wrong_weight_layout_raises(self, w0_shape, w1_shape, wo_shape, expected_msg):
    """Verifies _moe_body rejects non-3D or non-(num_experts // num_ep) local expert weights."""
    inputs, logits = jnp.ones((4, 4, 16), jnp.float32), jnp.ones((4, 4, 4), jnp.float32)
    w0 = jnp.ones(w0_shape, jnp.float32)
    w1 = jnp.ones(w1_shape, jnp.float32)
    wo = jnp.ones(wo_shape, jnp.float32)
    layer = self._make_layer(num_ep=2, expert_replicas_per_device=1)
    with self._cpu_ep_runtime(layer, num_ep=2):
      with self.assertRaisesRegex(ValueError, expected_msg):
        layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

  def test_unsharded_decode_batch_rejected(self):
    inputs, logits = jnp.ones((1, 4, 16), jnp.float32), jnp.ones((1, 4, 4), jnp.float32)
    w0, w1, wo = jnp.ones((2, 16, 8)), jnp.ones((2, 16, 8)), jnp.ones((2, 8, 16))
    layer = self._make_layer(num_ep=2, expert_replicas_per_device=1)
    with mock.patch.object(
        jax,
        "shard_map",
        side_effect=lambda fn, **_kw: lambda *a: jax.vmap(lambda _: fn(*a), axis_name="expert")(jnp.arange(2)),
    ):
      with self.assertRaisesRegex(ValueError, "is_batch_sharded_by_expert=True"):
        layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

  def test_routed_bias_updates_match_r0(self):
    """Verifies routed_bias_update_rate > 0 emits logical expert bias updates identical to R=0."""
    inputs, logits, w0, w1, wo, cotangent = self._sample_tensors(seed=7)
    logits = logits.at[:, :, 0].add(8.0).at[:, :, 1].add(7.0)
    ds_kw = {
        "decoder_block": "deepseek",
        "routed_score_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "n_routing_groups": -1,
        "topk_routing_group": -1,
        "routed_bias": True,
        "routed_bias_update_rate": 1e-3,
    }

    def run_arm(replicas):
      layer = self._make_layer(expert_replicas_per_device=replicas, expert_replica_max_send=2, **ds_kw)
      cap = {}
      with self._cpu_ep_runtime(layer, cotangent=cotangent, captured=cap):
        out, _, bias = layer.sparse_matmul(inputs, logits, logits, w0, w1, wo, None, None, None)
      return out, bias, cap

    out_ref, bias_ref, cap_base = run_arm(0)
    out_rep, bias_rep, cap_rep = run_arm(1)
    self.assertIsNotNone(bias_ref)
    self.assertGreater(int(np.asarray(cap_rep["local_group_sizes"])[1, -1]), 0)
    np.testing.assert_allclose(np.asarray(bias_rep), np.asarray(bias_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for g_rep, g_ref in zip(cap_rep["grads"], cap_base["grads"]):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  def test_remat_and_multi_branch_trace_parity_with_r0(self):
    """Verifies RouteMetadata.replica_plan PyTree flattening and remat + lax.cond parity vs R=0."""
    plan = mrp.compute_plan_iter(jnp.asarray([[20, 2, 1, 1], [10, 1, 1, 1]], jnp.int32), 2, 1, 1)
    rm = moe.RouteMetadata(
        0, jnp.arange(4, dtype=jnp.int32), jnp.asarray([12, 12], jnp.int32), jnp.ones((2, 2), jnp.int32), plan
    )
    leaves, treedef = jax.tree_util.tree_flatten(rm)
    self.assertLen(leaves, 9)
    rt = jax.jit(lambda f, m: jax.lax.cond(f, lambda z: z, lambda z: z, m))(
        jnp.asarray(True), jax.tree_util.tree_unflatten(treedef, leaves)
    )
    np.testing.assert_array_equal(np.asarray(rt.replica_plan.route_table), np.asarray(plan.route_table))

    inputs, logits, w0, w1, wo, cotangent = self._sample_tensors(seed=19)
    forced_a = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [0, 3]],
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )
    alt_forced = jnp.array([[[2, 3], [2, 3], [2, 3], [0, 2]], [[2, 3], [2, 3], [2, 3], [1, 3]]], dtype=jnp.int32)

    def wrap_remat_and_cond(fn, s_args):
      @jax.checkpoint
      def ckpt_shard(x_, g_, w0_, w1_, wo_, forced_main):
        return jax.lax.cond(
            jnp.bool_(True),
            lambda _: fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], forced_main, *s_args[12:]),
            lambda _: fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], alt_forced, *s_args[12:]),
            operand=None,
        )

      return ckpt_shard(s_args[0], s_args[1], s_args[3], s_args[4], s_args[5], s_args[11])

    def run_arm(replicas):
      layer = self._make_layer(expert_replicas_per_device=replicas, expert_replica_max_send=2)
      cap = {}
      with self._cpu_ep_runtime(layer, cotangent=cotangent, captured=cap, wrap_shard_fn=wrap_remat_and_cond):
        out, _, _ = layer.sparse_matmul(
            inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced_a
        )
      return out, cap["grads"]

    out_ref, grads_ref = run_arm(0)
    out_rep, grads_rep = run_arm(1)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for g_rep, g_ref in zip(grads_rep, grads_ref):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  def test_token_mask_padded_rows_excluded_from_planner_and_replica_slots(self):
    """Verifies token_mask padded tokens add 0 to planner counts, avoid replica slots, and match R=0."""
    inputs, base_logits, w0, w1, wo, cotangent = self._sample_tensors(seed=101)
    # Shard 0 (rows 0..1) has 2 valid tokens routing to (2, 0) plus 6 padded tokens whose round-robin
    # dummy_indices include hot expert 2; shard 1 (rows 2..3) has 7 valid tokens routing to (2, 3) + 1 pad.
    token_mask_mixed = jnp.array(
        [
            [True, True, False, False],
            [False, False, False, False],
            [True, True, True, True],
            [True, True, True, False],
        ],
        dtype=jnp.bool_,
    )
    logits_mixed = (
        base_logits.at[:2, :, 2]
        .set(12.0)
        .at[:2, :, 0]
        .set(9.0)
        .at[:2, :, 1]
        .set(-8.0)
        .at[:2, :, 3]
        .set(-8.0)
        .at[2:, :, 2]
        .set(12.0)
        .at[2:, :, 3]
        .set(9.0)
        .at[2:, :, 0]
        .set(-8.0)
        .at[2:, :, 1]
        .set(-8.0)
    )
    # Also test a tail-padded batch where shard 1 holds 0 valid tokens while shard 0 routes to shard 1's (2, 3).
    token_mask_tail = jnp.array(
        [[True, True, True, True], [True, True, True, True], [False, False, False, False], [False, False, False, False]],
        dtype=jnp.bool_,
    )
    logits_tail = base_logits.at[:2, :, 2].set(12.0).at[:2, :, 3].set(9.0).at[:2, :, :2].set(-8.0)

    orig_compute_plan = mrp.compute_plan_iter

    def check_case(logits, token_mask, expected_counts, expected_rep_gs):
      # Poison padded inputs with large non-zero values to verify they never leak into outputs or grads.
      padded_inputs = jnp.where(token_mask[..., None], inputs, jnp.float32(500.0))
      seen_counts = []

      def spy_plan(counts_src, num_ep, num_replicas, max_send):
        jax.debug.callback(lambda c: seen_counts.append(np.asarray(c)), counts_src)
        return orig_compute_plan(counts_src, num_ep, num_replicas, max_send)

      def run_arm(replicas):
        layer = self._make_layer(
            decoder_block="deepseek",
            expert_replicas_per_device=replicas,
            expert_replica_max_send=2,
            load_balance_loss_weight=0.01,
            routed_bias=True,
            routed_bias_update_rate=1e-3,
        )
        cap = {}
        ctx = (
            mock.patch.object(mrp, "compute_plan_iter", side_effect=spy_plan)
            if replicas > 0
            else contextlib.nullcontext()
        )
        with ctx, self._cpu_ep_runtime(layer, cotangent=cotangent, captured=cap):
          out, lb_loss, bias = layer.sparse_matmul(
              padded_inputs, logits, None, w0, w1, wo, None, None, None, token_mask=token_mask
          )
        return out, lb_loss, bias, cap

      out_ref, lb_ref, bias_ref, cap_ref = run_arm(0)
      out_rep, lb_rep, bias_rep, cap_rep = run_arm(1)

      # 1. Planner [EP, E] counts exclude all token_mask=False padded tokens (exact 0s on tail-padded shard).
      self.assertNotEmpty(seen_counts)
      for shard_counts in seen_counts:
        np.testing.assert_array_equal(shard_counts, np.array(expected_counts, dtype=np.int32))

      # 2. Replica slots [L, L+R) hold ONLY the planned valid tokens (zero dummy padded tokens).
      local_gs = np.asarray(cap_rep["local_group_sizes"])
      np.testing.assert_array_equal(local_gs[:, 2], np.array(expected_rep_gs, dtype=np.int32))

      # 3. Forward output, masked aux loss, routed-bias update, and all 5 gradients match R=0; padded positions are 0.
      pad_mask_np = ~np.asarray(token_mask)
      np.testing.assert_array_equal(np.asarray(out_rep)[pad_mask_np], 0.0)
      np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.asarray(lb_rep), np.asarray(lb_ref), rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(np.asarray(bias_rep), np.asarray(bias_ref), rtol=1e-6, atol=1e-6)
      dx_rep, dlogits_rep = np.asarray(cap_rep["grads"][0]), np.asarray(cap_rep["grads"][1])
      np.testing.assert_array_equal(dx_rep[pad_mask_np], 0.0)
      np.testing.assert_array_equal(dlogits_rep[pad_mask_np], 0.0)
      for name, g_rep, g_ref in zip(("dx", "dlogits", "dw0", "dw1", "dwo"), cap_rep["grads"], cap_ref["grads"]):
        np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5, err_msg=name)

    for case_name, logits_c, mask_c, exp_counts, exp_rep_gs in (
        ("mixed_pad_with_dummy_hitting_hot_expert", logits_mixed, token_mask_mixed, [[2, 0, 2, 0], [0, 0, 7, 7]], [2, 0]),
        ("tail_shard_all_padded", logits_tail, token_mask_tail, [[0, 0, 8, 8], [0, 0, 0, 0]], [8, 0]),
    ):
      with self.subTest(case=case_name):
        check_case(logits_c, mask_c, exp_counts, exp_rep_gs)


@pytest.mark.tpu_only
class PlannerDrivenReplicaWeightsTpuTest(parameterized.TestCase):
  """Runs `compute_plan_iter` + `extend_with_replicas` under `jax.shard_map` with real TPU `ragged_all_to_all`."""

  @parameterized.named_parameters(
      ("r1_maxsend1_f32", 1, 1, jnp.float32, False),
      ("r2_maxsend2_f32", 2, 2, jnp.float32, False),
      ("r3_maxsend4_f32", 3, 4, jnp.float32, False),
      ("r2_maxsend3_bf16", 2, 3, jnp.bfloat16, False),
      ("r2_maxsend2_tuple_axis_f32", 2, 2, jnp.float32, True),
      ("r2_maxsend3_tuple_axis_bf16", 2, 3, jnp.bfloat16, True),
  )
  def test_compute_plan_iter_and_real_ragged_all_to_all_match_reference(
      self, num_replicas, max_send, compute_dtype, use_tuple_axis
  ):
    """Checks forward and backward parity of planner-generated plans on 1D and tuple-axis TPU meshes."""
    devs = jax.devices()
    if len(devs) < 2:
      self.skipTest(f"compute_plan_iter requires >= 2 TPU devices, got {len(devs)}.")
    if use_tuple_axis and (len(devs) < 4 or len(devs) % 2 != 0):
      self.skipTest(f"tuple-axis mesh requires an even >= 4 TPU device count, got {len(devs)}.")
    self.assertIsNot(jax.lax.ragged_all_to_all, moe_replica_test_utils.emulated_ragged_all_to_all)
    num_ep = len(devs)
    local = 4
    num_experts = num_ep * local
    rng = np.random.default_rng(100 + num_replicas * 10 + max_send + int(use_tuple_axis))
    counts = rng.integers(1, 5, size=(num_ep, num_experts), dtype=np.int32)
    counts[:, :local] += 80
    if num_ep >= 4:
      counts[:, local : 2 * local] += 50

    plan = mrp.compute_plan_iter(jnp.asarray(counts), num_ep, num_replicas, max_send)
    self.assertGreater(int((np.asarray(plan.rep_home) >= 0).sum()), 0)

    w_np = rng.standard_normal((num_ep, local, 16, 128)).astype(np.float32)
    g_np = rng.standard_normal((num_ep, local + num_replicas, 16, 128)).astype(np.float32)
    if use_tuple_axis:
      mesh = Mesh(np.asarray(devs).reshape(2, num_ep // 2), ("context", "expert"))
      ep_axis = ("context", "expert")
    else:
      mesh = Mesh(np.asarray(devs), ("expert",))
      ep_axis = "expert"
    ep_spec = P(ep_axis)
    counts_sharded = jax.device_put(jnp.asarray(counts), jax.sharding.NamedSharding(mesh, ep_spec))
    w_sharded = jax.device_put(jnp.asarray(w_np), jax.sharding.NamedSharding(mesh, ep_spec))
    g_sharded = jax.device_put(jnp.asarray(g_np, dtype=compute_dtype), jax.sharding.NamedSharding(mesh, ep_spec))

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(ep_spec, ep_spec, ep_spec),
        out_specs=(ep_spec, ep_spec, ep_spec),
    )
    def step(counts_shard, w_shard, g_shard):
      counts_src = jax.lax.all_gather(counts_shard[0], axis_name=ep_axis)
      in_shard_plan = mrp.compute_plan_iter(counts_src, num_ep, num_replicas, max_send)
      me = jax.lax.axis_index(ep_axis)
      out_local, vjp_fn = jax.vjp(
          lambda w: mrw.extend_with_replicas(w, in_shard_plan, ep_axis, num_ep, compute_dtype),
          w_shard[0],
      )
      (dw_local,) = vjp_fn(g_shard[0])
      return out_local[None], dw_local[None], in_shard_plan.route_table[me][None]

    out, dw, shard_routes = step(counts_sharded, w_sharded, g_sharded)
    np.testing.assert_array_equal(np.asarray(shard_routes), np.asarray(plan.route_table))
    self.assertEqual(out.dtype, compute_dtype)
    self.assertEqual(dw.dtype, jnp.float32)

    w_ref = np.asarray(jnp.asarray(w_np, dtype=compute_dtype), dtype=np.float32)
    g_ref = np.asarray(jnp.asarray(g_np, dtype=compute_dtype), dtype=np.float32)
    reps_ref, dw_rep_ref = moe_replica_test_utils.replica_reference(w_ref, g_ref[:, local:], plan)
    np.testing.assert_array_equal(np.asarray(out, dtype=np.float32), np.concatenate([w_ref, reps_ref], axis=1))
    tol = 1e-2 if compute_dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(np.asarray(dw), g_ref[:, :local] + dw_rep_ref, rtol=tol, atol=tol)


_FP8_FULL_KW = {
    "quantization": "fp8_full",
    "use_qwix_quantization": True,
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "quantize_router_proj": False,
}


def _group_bounds(group_sizes, group_offset, n_rows):
  offset = 0 if group_offset is None else jnp.asarray(group_offset, jnp.int32).reshape(())
  starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes.astype(jnp.int32))])
  return starts, offset, jnp.arange(n_rows, dtype=jnp.int32)


def _ref_gmm_v2(
    lhs, rhs, group_sizes, rhs_scale=None, rhs_bias=None, partial_sum=None, group_offset=None, lhs_scale=None, **kw
):
  """Pure-JAX stand-in for tokamax gmm_v2 (`rhs_scale` [G, 1, 1, N])."""
  assert rhs_bias is None and not kw.get("out_is_3d")
  rhs = rhs.swapaxes(1, 2) if kw.get("transpose_rhs") else rhs
  x, post_scale = lhs.astype(jnp.float32), None if lhs_scale is None else lhs_scale.reshape(())
  if kw.get("maybe_quantize_lhs", True) and jnp.dtype(rhs.dtype).itemsize == 1 and jnp.dtype(lhs.dtype).itemsize > 1:
    if post_scale is None:
      post_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True), 1e-12) / 448.0
    x = jnp.clip(x / post_scale, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)
  starts, offset, rows = _group_bounds(group_sizes, group_offset, x.shape[0])
  out = jnp.zeros((x.shape[0], rhs.shape[2]), jnp.float32)
  for g in range(rhs.shape[0]):
    proj = jnp.dot(x, rhs[g].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    if rhs_scale is not None:
      proj = proj * rhs_scale[g, 0, 0][None, :]
    out = jnp.where(((rows >= starts[offset + g]) & (rows < starts[offset + g + 1]))[:, None], proj, out)
  out = out * post_scale if post_scale is not None else out
  out = out + partial_sum.astype(jnp.float32) if partial_sum is not None else out
  return out.astype(kw.get("preferred_element_type") or lhs.dtype)


def _ref_tgmm_v2(lhs, rhs, group_sizes, num_actual_groups, rhs_scale=None, partial_sum=None, group_offset=None, **kw):
  """Pure-JAX stand-in for tokamax tgmm_v2 (`out[g] = lhs[rows_g]^T @ rhs[rows_g] * rhs_scale`)."""
  x, y = lhs.astype(jnp.float32), rhs.astype(jnp.float32)
  starts, offset, rows = _group_bounds(group_sizes, group_offset, x.shape[0])
  out = jnp.stack(
      [
          jnp.dot(
              jnp.where(((rows >= starts[offset + g]) & (rows < starts[offset + g + 1]))[:, None], x, 0.0).T,
              y,
              precision=jax.lax.Precision.HIGHEST,
          )
          for g in range(num_actual_groups)
      ]
  )
  out = out * rhs_scale.reshape(1, 1, -1) if rhs_scale is not None else out
  out = out + partial_sum.astype(jnp.float32) if partial_sum is not None else out
  return out.astype(kw.get("preferred_element_type") or jnp.float32)


def _fp8_cal_kw(calibration):
  w_act, bwd = {
      "absmax": ("absmax", "absmax"),
      "absmax_fixed_bwd": ("absmax", "fixed,0.00125"),
      "fixed": ("fixed,-224,224", "fixed,0.00125"),
  }[calibration]
  return {
      "weight_quantization_calibration_method": w_act,
      "act_quantization_calibration_method": w_act,
      "bwd_quantization_calibration_method": bwd,
  }


class RoutedMoEReplicationFp8Test(parameterized.TestCase):
  """CPU tests for Qwix fp8_full (dynamic & static, 2D FSDP x EP, token_mask) and fp8_moe_fake_quant."""

  _make_layer = RoutedMoEReplicationWiringTest._make_layer  # pylint: disable=protected-access
  _cpu_ep_runtime = RoutedMoEReplicationWiringTest._cpu_ep_runtime  # pylint: disable=protected-access
  _sample_tensors = RoutedMoEReplicationWiringTest._sample_tensors  # pylint: disable=protected-access

  def _run(self, layer, tensors, forced, quantized, *, num_fsdp=1, token_mask=None):
    """Runs sparse_matmul fwd+bwd with optional Qwix gmm rule and pure-JAX gmm_v2/tgmm_v2."""
    inputs, logits, w0, w1, wo, cotangent = tensors
    captured = {}
    with contextlib.ExitStack() as stack:
      gmm_impl = None
      if quantized:
        rule = quantizations.get_fp8_full_qwix_rule_w_sparsity(layer.config)[-1]
        stack.enter_context(mock.patch.object(qpl, "get_current_rule", lambda op: rule if op == "gmm" else None))
        stack.enter_context(mock.patch.object(mblx_ops.gmm_v2, "gmm_v2", _ref_gmm_v2))
        stack.enter_context(mock.patch.object(mblx_ops.tgmm_v2, "tgmm_v2", _ref_tgmm_v2))
        gmm_impl = mblx_ops.gmm
      stack.enter_context(
          self._cpu_ep_runtime(
              layer, num_ep=2, num_fsdp=num_fsdp, cotangent=cotangent, captured=captured, gmm_impl=gmm_impl
          )
      )
      out, _, _ = layer.sparse_matmul(
          inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced, token_mask=token_mask
      )
    return out, captured

  def _inputs(self, seed=7, batch=4):
    """Generates skewed 4-expert routing inputs with 4x larger weights on hot expert 0."""
    forced = jnp.broadcast_to(jnp.array([[0, 1], [0, 1], [0, 1], [0, 2]], dtype=jnp.int32)[None], (batch, 4, 2))
    return self._sample_tensors(seed, batch=batch, expert0_mult=4.0), forced

  def _assert_fp8_parity(self, out_rep, cap_rep, out_ref, cap_ref, out_hp, cap_hp, *, exact_dw):
    """Asserts FP8 output/grad parity with R=0 and separation from BF16/FP32."""
    self.assertGreater(float(np.max(np.abs(np.asarray(out_ref) - np.asarray(out_hp)))), 1e-4)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for name, g_rep, g_ref, g_hp in zip(
        ("dx", "dlogits", "dw0", "dw1", "dwo"), cap_rep["grads"], cap_ref["grads"], cap_hp["grads"]
    ):
      with self.subTest(grad=name):
        g_rep_a, g_ref_a, g_hp_a = (np.asarray(g, np.float64) for g in (g_rep, g_ref, g_hp))
        if exact_dw or not name.startswith("dw"):
          np.testing.assert_allclose(g_rep_a, g_ref_a, rtol=1e-5, atol=1e-5)
        else:
          self.assertLessEqual(
              np.linalg.norm(g_rep_a - g_ref_a) / np.linalg.norm(g_ref_a),
              np.linalg.norm(g_ref_a - g_hp_a) / np.linalg.norm(g_hp_a),
          )

  @parameterized.named_parameters(
      ("absmax", "absmax", False, 1, True),
      ("absmax_fixed_bwd", "absmax_fixed_bwd", False, 1, True),
      ("fixed", "fixed", False, 1, True),
      ("fixed_2d_skewed", "fixed", False, 2, True),
      ("fixed_2d_balanced_no_send", "fixed", False, 2, False),
  )
  def test_fp8_full_replication_matches_r0(self, calibration, disable_cw, num_fsdp, skewed):
    """Covers dynamic/static calibration, channelwise/per-tensor, 1D/2D meshes, and zero-send."""
    tensors, forced = self._inputs(seed=7 if num_fsdp == 1 else 13, batch=4 * num_fsdp)
    if not skewed:
      forced = jnp.broadcast_to(jnp.array([[0, 2], [1, 3], [0, 2], [1, 3]], dtype=jnp.int32)[None], (4 * num_fsdp, 4, 2))
    fp8_kw = {
        **_FP8_FULL_KW,
        "qwix_disable_channelwise_axes": disable_cw,
        "ici_expert_parallelism": 2,
        "ici_fsdp_parallelism": num_fsdp if num_fsdp > 1 else 4,
        **_fp8_cal_kw(calibration),
    }

    def _build(replicas, quant=True):
      layer = self._make_layer(
          num_ep=2,
          expert_replicas_per_device=replicas,
          expert_replica_max_send=2,
          **(fp8_kw if quant else {}),
      )
      if num_fsdp > 1:

        def _axes_2d(axes):
          return P(
              *(
                  ("fsdp", "expert") if a == "activation_batch" else ("expert" if a in ("exp", "exp_with_fsdp") else None)
                  for a in axes
              )
          )

        layer._logical_to_mesh_axes = _axes_2d  # pylint: disable=protected-access
      return layer

    out_ref, cap_ref = self._run(_build(0), tensors, forced, True, num_fsdp=num_fsdp)
    out_rep, cap_rep = self._run(_build(1), tensors, forced, True, num_fsdp=num_fsdp)
    out_hp, cap_hp = self._run(_build(0, False), tensors, forced, False, num_fsdp=num_fsdp)
    rep_gs = int(np.asarray(cap_rep["local_group_sizes"])[..., -1].sum())
    if not skewed:
      self.assertEqual(rep_gs, 0)
    else:
      self.assertGreater(rep_gs, 0)
    self._assert_fp8_parity(out_rep, cap_rep, out_ref, cap_ref, out_hp, cap_hp, exact_dw=calibration != "absmax")

  def test_fp8_moe_fake_quant_replication_matches_r0(self):
    """fp8_moe_fake_quant rounds weights before sparse_matmul, so dense replication matches R=0."""
    (inputs, _, w0, w1, wo, cotangent), forced = self._inputs(seed=21)

    def _run_call(replicas, fake_quant=True):
      layer = self._make_layer(
          expert_replicas_per_device=replicas, expert_replica_max_send=2, fp8_moe_fake_quant=fake_quant
      )
      layer.wi_0[...], layer.wi_1[...], layer.wo[...] = w0, w1, wo
      cap = {}
      with self._cpu_ep_runtime(layer, cotangent=cotangent, captured=cap):
        out, _, _ = layer(inputs, forced_routed_experts=forced)
      return out, cap

    out_hp, _ = _run_call(0, False)
    out_ref, cap_ref = _run_call(0)
    out_rep, cap_rep = _run_call(1)
    self.assertGreater(float(np.max(np.abs(np.asarray(out_ref) - np.asarray(out_hp)))), 1e-4)
    self.assertGreater(int(np.asarray(cap_rep["local_group_sizes"])[1, -1]), 0)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for g_rep, g_ref in zip(cap_rep["grads"], cap_ref["grads"]):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  def test_quantized_runtime_requires_gmm_v2_on_tpu(self):
    """Raises NotImplementedError when a Qwix gmm rule is active without tokamax gmm_v2 on TPU."""
    (inputs, logits, w0, w1, wo, _), forced = self._inputs()
    layer = self._make_layer(expert_replicas_per_device=1, expert_replica_max_send=2)
    rule = qwix.QtRule(module_path=".*", weight_qtype=jnp.float8_e4m3fn, op_names=("gmm",))
    object.__setattr__(layer.config, "use_qwix_quantization", True)
    with (
        mock.patch.object(qpl, "get_current_rule", lambda op: rule if op == "gmm" else None),
        self._cpu_ep_runtime(layer),
    ):
      with self.assertRaisesRegex(NotImplementedError, "gmm_v2"):
        layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)

  def test_quantized_weight_gather_requires_bwd_qtype(self):
    """Defense-in-depth: sparse_matmul rejects weight gathers without bwd_qtype (needed for QAG)."""
    (inputs, logits, w0, w1, wo, _), forced = self._inputs(batch=8)
    layer = self._make_layer(
        num_ep=2,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        **_FP8_FULL_KW,
        **_fp8_cal_kw("fixed"),
        shard_embed_moe_on_fsdp=True,
        ici_expert_parallelism=2,
        ici_fsdp_parallelism=2,
    )
    layer._logical_to_mesh_axes = lambda axes: P(  # pylint: disable=protected-access
        *(
            ("fsdp", "expert")
            if a == "activation_batch"
            else ("fsdp" if a == "embed_moe" else ("expert" if a in ("exp", "exp_with_fsdp") else None))
            for a in axes
        )
    )
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        weight_calibration_method="fixed,-224,224",
        bwd_qtype=None,
        op_names=("gmm",),
    )
    with (
        mock.patch.object(qpl, "get_current_rule", lambda op: rule if op == "gmm" else None),
        self._cpu_ep_runtime(layer, num_ep=2, num_fsdp=2, gmm_impl=mblx_ops.gmm),
    ):
      with self.assertRaisesRegex(NotImplementedError, "requires fixed weight calibration and bwd_qtype"):
        layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)

  @parameterized.named_parameters(
      ("no_qag_distinct_plans", False, "antisym", 16),
      ("qag_active_synced_replica", True, "asym_active", 10),
      ("qag_antisymmetric_batches", True, "antisym", 0),
  )
  def test_fp8_full_2d_fsdp_ep_distinct_batches_match_r0(self, shard_embed_fsdp, pattern, expected_rep_tokens):
    """Verifies 2D FSDP x EP parity when FSDP slices have distinct token routing (with and without QAG)."""
    tensors = self._sample_tensors(seed=37, batch=8, expert0_mult=4.0)
    if pattern == "antisym":
      f0 = jnp.broadcast_to(jnp.array([[0, 1], [0, 1], [0, 1], [0, 2]], jnp.int32)[None], (4, 4, 2))
      f1 = jnp.broadcast_to(jnp.array([[2, 3], [2, 3], [2, 3], [2, 0]], jnp.int32)[None], (4, 4, 2))
    else:
      # Without cross-FSDP count sync, f0 replicates expert 0 (13 vs 11) while f1 replicates expert 1 (15 vs 9)
      # into the same ep=1 replica slot; synced counts [22, 26, 6, 10] replicate expert 1 on both slices (3 + 7 = 10).
      f0_ep0 = jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (2, 4, 2))
      f0_ep1 = jnp.array(
          [[[0, 2], [0, 2], [0, 2], [0, 2]], [[0, 2], [1, 3], [1, 3], [1, 3]]],
          dtype=jnp.int32,
      )
      f1_ep0 = jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (2, 4, 2))
      f1_ep1 = jnp.array(
          [[[0, 2], [1, 3], [1, 3], [1, 3]], [[1, 3], [1, 3], [1, 3], [1, 3]]],
          dtype=jnp.int32,
      )
      f0 = jnp.concatenate([f0_ep0, f0_ep1], axis=0)
      f1 = jnp.concatenate([f1_ep0, f1_ep1], axis=0)
    forced = jnp.concatenate([f0, f1], axis=0)
    fp8_kw = {
        **_FP8_FULL_KW,
        **_fp8_cal_kw("fixed"),
        "shard_embed_moe_on_fsdp": shard_embed_fsdp,
        "ici_expert_parallelism": 2,
        "ici_fsdp_parallelism": 2,
    }

    def _build(replicas, quant=True):
      layer = self._make_layer(
          num_ep=2,
          expert_replicas_per_device=replicas,
          expert_replica_max_send=2,
          **(fp8_kw if quant else {}),
      )
      layer._logical_to_mesh_axes = lambda axes: P(  # pylint: disable=protected-access
          *(
              ("fsdp", "expert")
              if a == "activation_batch"
              else ("fsdp" if a == "embed_moe" else ("expert" if a in ("exp", "exp_with_fsdp") else None))
              for a in axes
          )
      )
      return layer

    out_ref, cap_ref = self._run(_build(0), tensors, forced, True, num_fsdp=2)
    out_rep, cap_rep = self._run(_build(1), tensors, forced, True, num_fsdp=2)
    out_hp, cap_hp = self._run(_build(0, False), tensors, forced, False, num_fsdp=2)
    self.assertEqual(int(np.asarray(cap_rep["local_group_sizes"])[..., -1].sum()), expected_rep_tokens)
    self._assert_fp8_parity(out_rep, cap_rep, out_ref, cap_ref, out_hp, cap_hp, exact_dw=True)

  @parameterized.named_parameters(("absmax", "absmax"), ("fixed", "fixed"))
  def test_fp8_full_padded_tokens_stay_on_home_slots_and_match_r0(self, calibration):
    """Padded tokens (token_mask=False) stay out of FP8 planner counts and replica slots, matching R=0."""
    (inputs, base_logits, w0, w1, wo, cotangent), _ = self._inputs(seed=29)
    token_mask = jnp.array(
        [[False, False, False, False], [True, False, True, False], [True, True, True, True], [True, True, True, False]],
        dtype=jnp.bool_,
    )
    logits = base_logits.at[:2, :, 2].set(12.0).at[:2, :, 0].set(9.0).at[:2, :, 1].set(-8.0).at[:2, :, 3].set(-8.0)
    logits = logits.at[2:, :, 2].set(12.0).at[2:, :, 3].set(9.0).at[2:, :, 0].set(-8.0).at[2:, :, 1].set(-8.0)
    fp8_kw = {**_FP8_FULL_KW, **_fp8_cal_kw(calibration)}
    seen_counts, orig_plan = [], mrp.compute_plan_iter

    def _run_padded(reps, pad_val, quant=True):
      layer = self._make_layer(
          expert_replicas_per_device=reps,
          expert_replica_max_send=2,
          ici_expert_parallelism=2,
          **(fp8_kw if quant else {}),
      )
      padded = (jnp.where(token_mask[..., None], inputs, jnp.float32(pad_val)), logits, w0, w1, wo, cotangent)
      return self._run(layer, padded, None, quant, token_mask=token_mask)

    out_ref, cap_ref = _run_padded(0, 500.0)
    out_hp, cap_hp = _run_padded(0, 500.0, False)
    out_zero, cap_zero = _run_padded(1, 0.0)
    with mock.patch.object(
        mrp,
        "compute_plan_iter",
        side_effect=lambda c, *a: (jax.debug.callback(lambda x: seen_counts.append(np.asarray(x)), c), orig_plan(c, *a))[
            1
        ],
    ):
      out_rep, cap_rep = _run_padded(1, 500.0)
    self.assertNotEmpty(seen_counts)
    for c in seen_counts:
      np.testing.assert_array_equal(c, np.array([[2, 0, 2, 0], [0, 0, 7, 7]], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(cap_rep["local_group_sizes"])[:, 2], np.array([2, 0], dtype=np.int32))
    pad_mask = ~np.asarray(token_mask)
    np.testing.assert_array_equal(np.asarray(out_rep)[pad_mask], 0.0)
    np.testing.assert_array_equal(np.asarray(cap_rep["grads"][0])[pad_mask], 0.0)
    np.testing.assert_array_equal(np.asarray(cap_rep["grads"][1])[pad_mask], 0.0)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_zero), rtol=1e-5, atol=1e-5)
    for g_rep, g_zero in zip(cap_rep["grads"], cap_zero["grads"]):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_zero), rtol=1e-5, atol=1e-5)
    self._assert_fp8_parity(out_rep, cap_rep, out_ref, cap_ref, out_hp, cap_hp, exact_dw=calibration != "absmax")


@pytest.mark.tpu_only
class RoutedMoEReplicationFp8TpuTest(parameterized.TestCase):
  """End-to-end TPU tests exercising real Qwix interception + Pallas gmm_v2/tgmm_v2 with R>0 vs R=0."""

  _TILE_OVERRIDES = {
      f"{p}_tile_{s}_{d}": 256 if d == "mlp_dim" else 128
      for p in ("wi", "wo")
      for s in ("fwd", "dlhs", "drhs")
      for d in ("batch_seq", "embed_dim", "mlp_dim")
  }

  def _run_tpu(self, replicas, calibration, disable_cw, ici_ep, ici_fsdp, shard_embed_fsdp, forced, x, cotangent):
    """Compiles and executes quantized RoutedMoE fwd+bwd on TPU via Qwix + tokamax gmm_v2/tgmm_v2."""
    cfg = _init_moe_cfg(
        "moe_replication_fp8_tpu",
        **_FP8_FULL_KW,
        **self._TILE_OVERRIDES,
        **_fp8_cal_kw(calibration),
        base_emb_dim=256,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        dtype="bfloat16",
        per_device_batch_size=2,
        max_target_length=64,
        prefuse_moe_weights=False,
        ici_expert_parallelism=ici_ep,
        ici_fsdp_parallelism=ici_fsdp,
        shard_embed_moe_on_fsdp=shard_embed_fsdp,
        qwix_disable_channelwise_axes=disable_cw,
        expert_replicas_per_device=replicas,
        expert_replica_max_send=2,
    )
    mesh = Mesh(np.asarray(jax.devices()[: ici_ep * ici_fsdp]).reshape(ici_fsdp, ici_ep), ("fsdp", "expert"))
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=jnp.float8_e5m2,
        weight_calibration_method=cfg.weight_quantization_calibration_method,
        act_calibration_method=cfg.act_quantization_calibration_method,
        bwd_calibration_method=cfg.bwd_quantization_calibration_method,
        disable_channelwise_axes=cfg.qwix_disable_channelwise_axes,
        op_names=("gmm", "ragged_dot"),
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      layer = moe.RoutedMoE(
          config=cfg,
          num_experts=4,
          num_experts_per_tok=2,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=256,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.float32,
          rngs=nnx.Rngs(0),
      )
      layer = qwix.quantize_model(layer, qwix.QtProvider([rule]), x, forced_routed_experts=forced)
      nnx.pop(layer, nnx.Intermediate)

      @nnx.jit
      def step(m, h):
        def loss_fn(m_, h_):
          out, _, _ = m_(h_, forced_routed_experts=forced)
          return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

        (_, out), (m_grad, x_grad) = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(m, h)
        return out, x_grad, m_grad

      return step(layer, x)

  @parameterized.named_parameters(
      ("absmax_ep2", "absmax", False, 2, 1, False, "same"),
      ("fixed_pertensor_ep2", "fixed", True, 2, 1, False, "same"),
      ("fixed_2d_fsdp_ep_distinct_plans", "fixed", True, 2, 2, False, "antisym"),
      ("fixed_2d_fsdp_ep_qag_active_rep", "fixed", False, 2, 2, True, "asym_active"),
      ("fixed_2d_fsdp_ep_qag_antisym", "fixed", True, 2, 2, True, "antisym"),
  )
  def test_tpu_fp8_replication_parity_with_r0(self, calibration, disable_cw, ici_ep, ici_fsdp, shard_embed_fsdp, pattern):
    """Verifies Pallas gmm_v2 + tgmm_v2 fwd/bwd parity on TPU (including QAG + distinct FSDP batches)."""
    if len(jax.devices()) < ici_ep * ici_fsdp:
      self.skipTest(f"Requires >= {ici_ep * ici_fsdp} TPU devices, got {len(jax.devices())}.")
    total_batch = 2 * ici_ep * ici_fsdp
    k_x, k_c = jax.random.split(jax.random.PRNGKey(31), 2)
    x = jax.random.normal(k_x, (total_batch, 64, 256), jnp.bfloat16)
    cotangent = jax.random.normal(k_c, (total_batch, 64, 256), jnp.float32)
    if pattern == "antisym":
      half = total_batch // 2
      f0 = (
          jnp.broadcast_to(jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (64, 2))[None], (half, 64, 2))
          .at[:, -1, :]
          .set(jnp.array([2, 3], jnp.int32))
      )
      f1 = (
          jnp.broadcast_to(jnp.broadcast_to(jnp.array([[2, 3]], jnp.int32), (64, 2))[None], (total_batch - half, 64, 2))
          .at[:, -1, :]
          .set(jnp.array([0, 1], jnp.int32))
      )
      forced = jnp.concatenate([f0, f1], axis=0)
    elif pattern == "asym_active":
      # Without cross-FSDP count sync, f0 picks expert 0 (198 vs 186) while f1 picks expert 1 (226 vs 158) for
      # ep=1's replica slot; synced counts [356, 412, 100, 156] actively replicate expert 1 on both FSDP slices.
      ep0_rows = jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (2, 64, 2))
      f0_ep1 = (
          jnp.broadcast_to(jnp.array([[1, 3]], jnp.int32), (2, 64, 2)).at[:, :35, :].set(jnp.array([0, 2], jnp.int32))
      )
      f1_ep1 = (
          jnp.broadcast_to(jnp.array([[1, 3]], jnp.int32), (2, 64, 2)).at[:, :15, :].set(jnp.array([0, 2], jnp.int32))
      )
      forced = jnp.concatenate([ep0_rows, f0_ep1, ep0_rows, f1_ep1], axis=0)
    else:
      forced = (
          jnp.broadcast_to(jnp.broadcast_to(jnp.array([[0, 1]], jnp.int32), (64, 2))[None], (total_batch, 64, 2))
          .at[:, -1, :]
          .set(jnp.array([2, 3], jnp.int32))
      )
    args = (calibration, disable_cw, ici_ep, ici_fsdp, shard_embed_fsdp, forced, x, cotangent)
    out_ref, dx_ref, mg_ref = self._run_tpu(0, *args)
    out_rep, dx_rep, mg_rep = self._run_tpu(1, *args)
    tol = 2e-2 if calibration == "fixed" else 3e-2
    np.testing.assert_allclose(np.asarray(out_rep, np.float32), np.asarray(out_ref, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dx_rep, np.float32), np.asarray(dx_ref, np.float32), rtol=tol, atol=tol)
    for attr in ("wi_0", "wi_1", "wo"):
      g_rep, g_ref = np.asarray(getattr(mg_rep, attr)[...], np.float64), np.asarray(
          getattr(mg_ref, attr)[...], np.float64
      )
      self.assertLess(
          np.linalg.norm(g_rep - g_ref) / np.maximum(np.linalg.norm(g_ref), 1e-12),
          2e-2 if calibration == "fixed" else 8e-2,
          msg=attr,
      )


if __name__ == "__main__":
  absltest.main()

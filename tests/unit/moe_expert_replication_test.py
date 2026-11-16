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

"""CPU tests for RoutedMoE online hot-expert replication wiring."""

import contextlib
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import moe_replica_planner as mrp
from maxtext.layers import moe_replica_weights as mrw
from tests.utils.test_helpers import get_test_config_path


def _emulated_ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name
):  # pylint: disable=too-many-positional-arguments
  """CPU-compatible ragged_all_to_all emulation via all_gather under jax.vmap."""
  del recv_sizes
  me = jax.lax.axis_index(axis_name)
  ops = jax.lax.all_gather(operand, axis_name)
  in_off = jax.lax.all_gather(input_offsets, axis_name)[:, me]
  size = jax.lax.all_gather(send_sizes, axis_name)[:, me]
  out_off = jax.lax.all_gather(output_offsets, axis_name)[:, me]
  rows = jnp.arange(output.shape[0])
  out = output
  for i in range(ops.shape[0]):
    hit = (rows >= out_off[i]) & (rows < out_off[i] + size[i])
    src = jnp.clip(in_off[i] + rows - out_off[i], 0, operand.shape[0] - 1)
    out = jnp.where(hit.reshape((-1,) + (1,) * (out.ndim - 1)), ops[i][src], out)
  return out


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


def _spec_shards_dim0_on_expert(spec):
  if spec is None:
    return False
  parts = spec.partitions if isinstance(spec, P) else tuple(spec)
  return bool(parts and parts[0] is not None and "expert" in (parts[0] if isinstance(parts[0], tuple) else (parts[0],)))


def _call_moe_body_return_weights(fn, s_args):
  """Calls inner _moe_body with return_weights=True to exercise the chunk-wgrad guardrail.

  _moe_body is nested inside sparse_matmul and only receives return_weights=True under
  use_ring_of_experts=True (which fails earlier in route()), so closure lookup reaches it directly.
  """
  moe_body = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))["_moe_body"]
  return moe_body(*s_args[:12], return_weights=True, per_expert_scale=s_args[12])[:6]


class RoutedMoEReplicationWiringTest(parameterized.TestCase):
  """CPU unit tests exercising RoutedMoE hot-expert replication paths directly."""

  def _make_layer(self, *, num_ep=2, num_experts=4, num_experts_per_tok=2, emb_dim=16, mlp_dim=8, **overrides):
    """Builds a float32 RoutedMoE layer configured for multi-shard EP emulation on CPU."""
    kwargs = {
        "run_name": "moe_replication_wiring_cpu",
        "base_output_directory": "/tmp/moe_expert_replication_test",
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
        "base_emb_dim": emb_dim,
        "base_moe_mlp_dim": mlp_dim,
        "base_mlp_dim": mlp_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "per_device_batch_size": 2,
        "max_target_length": 4,
        "load_balance_loss_weight": 0.0,
        "routed_bias": False,
        "compile_topology": "v5e-8",
        "compile_topology_num_slices": 1,
        "ici_expert_parallelism": max(num_ep, 2),
        "ici_fsdp_parallelism": 8 // max(num_ep, 2),
    }
    kwargs.update(overrides)
    cfg = pyconfig.initialize([None, get_test_config_path()], **kwargs)
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
  def _cpu_ep_runtime(self, layer, *, num_ep=2, cotangent=None, captured=None, wrap_shard_fn=None):
    """Emulates multi-shard EP execution (and SPMD VJP when cotangent is given) on 1 CPU device."""
    orig_hardware = layer.config.hardware
    shard_gs_box = []

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
      del mesh, check_vma
      in_axes = tuple(0 if _spec_shards_dim0_on_expert(s) else None for s in in_specs)

      def run_shard(s_args):
        shard_gs_box.clear()
        outs = wrap_shard_fn(fn, s_args) if wrap_shard_fn is not None else fn(*s_args)
        return outs, (shard_gs_box[0] if shard_gs_box else None)

      def wrapped(*args):
        split_args = [
            jnp.reshape(a, (num_ep, a.shape[0] // num_ep, *a.shape[1:])) if (ax == 0 and a is not None) else a
            for a, ax in zip(args, in_axes)
        ]
        if cotangent is not None:
          cot_shards = jnp.reshape(cotangent, (num_ep, cotangent.shape[0] // num_ep, *cotangent.shape[1:]))

          def per_shard_fwd_bwd(*all_s):
            *s_args, cot_s = all_s

            def diff_fn(x_, g_, w0_, w1_, wo_):
              outs, gs = run_shard((x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:]))
              return outs[0], (outs, gs)

            _, vjp_fn, (outs, gs) = jax.vjp(
                diff_fn,
                s_args[0],
                s_args[1],
                s_args[3],
                s_args[4],
                s_args[5],
                has_aux=True,
            )
            return outs, gs, vjp_fn(cot_s)

          raw_outs, local_gs, shard_grads = jax.vmap(per_shard_fwd_bwd, axis_name="expert", in_axes=(*in_axes, 0))(
              *split_args, cot_shards
          )
          if captured is not None:
            captured["grads"] = tuple(jnp.reshape(g, (num_ep * g.shape[1], *g.shape[2:])) for g in shard_grads)
        else:
          raw_outs, local_gs = jax.vmap(lambda *s_args: run_shard(s_args), axis_name="expert", in_axes=in_axes)(
              *split_args
          )
        if captured is not None:
          captured["local_group_sizes"] = local_gs
        return tuple(
            None
            if val is None
            else (jnp.reshape(val, (num_ep * val.shape[1], *val.shape[2:])) if _spec_shards_dim0_on_expert(s) else val[0])
            for val, s in zip(raw_outs, out_specs)
        )

      return wrapped

    def tracked_gmm(*args, **kwargs):
      if not shard_gs_box and wrap_shard_fn is None:
        gs = kwargs.get("group_sizes", args[2] if len(args) > 2 else None)
        if gs is not None:
          shard_gs_box.append(gs)
      return _pure_jax_gmm(*args, **kwargs)

    try:
      object.__setattr__(layer.config, "hardware", "tpu")
      with (
          mock.patch.object(jax, "shard_map", side_effect=fake_shard_map),
          mock.patch.object(jax.lax, "ragged_all_to_all", side_effect=_emulated_ragged_all_to_all),
          mock.patch.object(moe.mblx, "gmm", side_effect=tracked_gmm),
      ):
        yield
    finally:
      object.__setattr__(layer.config, "hardware", orig_hardware)

  def _sample_tensors(self, seed, num_experts=4):
    k_in, k_log, k_w0, k_w1, k_wo, k_cot = jax.random.split(jax.random.PRNGKey(seed), 6)
    return (
        jax.random.normal(k_in, (4, 4, 16), jnp.float32),
        jax.random.normal(k_log, (4, 4, num_experts), jnp.float32),
        jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.25,
        jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.25,
        jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.25,
        jax.random.normal(k_cot, (4, 4, 16), jnp.float32),
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
        physical_route_fn=lambda flat: (jnp.where((flat == 0) & (jnp.arange(flat.shape[0]) >= 6), 5, base_map[flat]), 6),
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

    def route_fn(flat):
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
            lambda _: fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], forced_main, s_args[12]),
            lambda _: fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], alt_forced, s_args[12]),
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


if __name__ == "__main__":
  absltest.main()

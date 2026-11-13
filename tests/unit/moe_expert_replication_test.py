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

"""CPU tests for hot-expert replication config flags and RoutedMoE wiring."""

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
import pydantic

from maxtext.configs import pyconfig
from maxtext.layers import initializers
from maxtext.layers import moe
from tests.utils.test_helpers import get_test_config_path


def _emulated_ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name):
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


def _pure_jax_gmm(
    lhs,
    rhs,
    group_sizes,
    preferred_element_type=jnp.float32,
    tiling=None,
    group_offset=None,
    existing_out=None,
    return_lhs=False,
    return_rhs=False,
    **_unused_kwargs,
):
  """Differentiable group-wise matmul matching megablox.gmm semantics on CPU."""
  del tiling
  offset = 0 if group_offset is None else jnp.asarray(group_offset, jnp.int32)
  starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes.astype(jnp.int32))])
  rows = jnp.arange(lhs.shape[0], dtype=jnp.int32)
  out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=preferred_element_type) if existing_out is None else existing_out
  for g in range(rhs.shape[0]):
    idx = offset + g
    in_group = (rows >= starts[idx]) & (rows < starts[idx + 1])
    proj = jnp.dot(lhs, rhs[g], precision=jax.lax.Precision.HIGHEST).astype(preferred_element_type)
    out = jnp.where(in_group[:, None], proj, out)
  if return_lhs and return_rhs:
    return out, lhs, rhs
  if return_lhs:
    return out, lhs
  if return_rhs:
    return out, rhs
  return out


def _spec_shards_dim0_on_expert(spec):
  """Returns True when PartitionSpec shards dim 0 across the expert mesh axis."""
  if spec is None:
    return False
  parts = spec.partitions if isinstance(spec, P) else tuple(spec)
  if not parts or parts[0] is None:
    return False
  dim0 = parts[0] if isinstance(parts[0], tuple) else (parts[0],)
  return "expert" in dim0


class ExpertReplicationConfigTest(absltest.TestCase):
  """Tests defaults and validation of expert_replica_* config flags."""

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_expert_replication_test",
        base_output_directory="/tmp/moe_expert_replication_test",
        enable_checkpointing=False,
        **kw,
    )

  def test_defaults_off(self):
    cfg = self._init()
    self.assertEqual(cfg.expert_replicas_per_device, 0)
    self.assertEqual(cfg.expert_replica_plan, "online")
    self.assertEqual(cfg.expert_replica_planner, "rounds")
    self.assertEqual(cfg.expert_replica_impl, "auto")
    self.assertEqual(cfg.expert_replica_move, "direct")
    self.assertFalse(cfg.expert_replica_identity)
    self.assertFalse(cfg.expert_replica_overhead_control)
    self.assertFalse(cfg.log_ra2a_required_rbf)

  def test_flags_parse(self):
    cfg = self._init(
        expert_replicas_per_device=4,
        expert_replica_max_send=3,
        expert_replica_plan="lagged",
        expert_replica_planner="greedy",
        expert_replica_impl="concat",
        expert_replica_move="gather",
        expert_replica_overhead_control=True,
        log_ra2a_required_rbf=True,
    )
    self.assertEqual(cfg.expert_replicas_per_device, 4)
    self.assertEqual(cfg.expert_replica_max_send, 3)
    self.assertEqual(cfg.expert_replica_plan, "lagged")
    self.assertEqual(cfg.expert_replica_planner, "greedy")
    self.assertEqual(cfg.expert_replica_impl, "concat")
    self.assertEqual(cfg.expert_replica_move, "gather")
    self.assertTrue(cfg.expert_replica_overhead_control)
    self.assertTrue(cfg.log_ra2a_required_rbf)

  def test_invalid_choices_rejected(self):
    for kw in (
        {"expert_replica_planner": "bogus"},
        {"expert_replica_impl": "bogus"},
        {"expert_replica_move": "bogus"},
        {"expert_replica_plan": "bogus"},
    ):
      with self.subTest(kw=kw):
        with self.assertRaises((pydantic.ValidationError, ValueError)):
          self._init(**kw)


class RoutedMoEReplicationWiringTest(parameterized.TestCase):
  """CPU unit tests exercising RoutedMoE hot-expert replication paths directly."""

  def _make_layer(self, num_ep=2, num_experts=4, num_experts_per_tok=2, emb_dim=16, mlp_dim=8, **overrides):
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
    }
    kwargs.update(overrides)
    cfg = pyconfig.initialize([None, get_test_config_path()], **kwargs)
    fake_mesh = types.SimpleNamespace(
        axis_names=("data", "fsdp", "expert", "tensor"),
        shape={"data": 1, "fsdp": 1, "expert": num_ep, "tensor": 1},
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
  def _cpu_ep_runtime(self, layer, num_ep=2, cotangent=None, force_return_weights=False, captured=None):
    """Emulates multi-shard EP execution (and SPMD VJP when cotangent is given) on 1 CPU device."""
    orig_hardware = layer.config.hardware
    shard_gs_box = []

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
      del mesh, check_vma
      in_axes = tuple(0 if _spec_shards_dim0_on_expert(s) else None for s in in_specs)

      def wrapped(*args):
        split_args = [
            jnp.reshape(a, (num_ep, a.shape[0] // num_ep, *a.shape[1:])) if (ax == 0 and a is not None) else a
            for a, ax in zip(args, in_axes)
        ]
        moe_body = None
        if force_return_weights:
          freevars = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
          moe_body = freevars["_moe_body"]

        def run_shard(s_args):
          shard_gs_box.clear()
          if force_return_weights:
            (
                x,
                logits,
                pre_bias_logits,
                w0,
                w1,
                wo,
                w0_bias,
                w1_bias,
                wo_bias,
                sharded_input_ids,
                rngs,
                forced_routed_experts,
                per_expert_scale,
            ) = s_args
            out, lb_loss, bias_updates, has_overflow, required_rbf, _, ret_weights = moe_body(
                x,
                logits,
                pre_bias_logits,
                w0,
                w1,
                wo,
                w0_bias,
                w1_bias,
                wo_bias,
                sharded_input_ids,
                rngs,
                forced_routed_experts,
                return_weights=True,
                per_expert_scale=per_expert_scale,
            )
            has_ov = jnp.reshape(jnp.asarray(has_overflow, dtype=jnp.bool_), (1,))
            gs = shard_gs_box[0] if shard_gs_box else None
            return (out, lb_loss, bias_updates, has_ov, jnp.bool_(False), required_rbf), ret_weights, gs
          outs = fn(*s_args)
          gs = shard_gs_box[0] if shard_gs_box else None
          return outs, None, gs

        if cotangent is not None:
          cot_shards = jnp.reshape(cotangent, (num_ep, cotangent.shape[0] // num_ep, *cotangent.shape[1:]))

          def per_shard_fwd_bwd(*all_s):
            *s_args, cot_s = all_s

            def diff_fn(x_, g_, w0_, w1_, wo_):
              rebuilt = (x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:])
              outs, ret_w, gs = run_shard(rebuilt)
              return outs[0], (outs, ret_w, gs)

            _, vjp_fn, (outs, ret_w, gs) = jax.vjp(
                diff_fn, s_args[0], s_args[1], s_args[3], s_args[4], s_args[5], has_aux=True
            )
            grads_s = vjp_fn(cot_s)
            return outs, ret_w, gs, grads_s

          raw_outs, ret_w, local_gs, shard_grads = jax.vmap(per_shard_fwd_bwd, axis_name="expert", in_axes=(*in_axes, 0))(
              *split_args, cot_shards
          )
          if captured is not None:
            captured["grads"] = tuple(jnp.reshape(g, (num_ep * g.shape[1], *g.shape[2:])) for g in shard_grads)
            captured["local_group_sizes"] = local_gs
            if force_return_weights:
              captured["ret_weights"] = ret_w
        else:
          raw_outs, ret_w, local_gs = jax.vmap(lambda *s_args: run_shard(s_args), axis_name="expert", in_axes=in_axes)(
              *split_args
          )
          if captured is not None:
            captured["local_group_sizes"] = local_gs
            if force_return_weights:
              captured["ret_weights"] = ret_w

        merged = []
        for val, spec in zip(raw_outs, out_specs):
          if val is None:
            merged.append(None)
          elif _spec_shards_dim0_on_expert(spec):
            merged.append(jnp.reshape(val, (num_ep * val.shape[1], *val.shape[2:])))
          else:
            merged.append(val[0])
        return tuple(merged)

      return wrapped

    def tracked_gmm(*args, **kwargs):
      if not shard_gs_box:
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

  def test_permute_with_physical_route_fn(self):
    """Verifies permute() remaps logical experts into EP*(L+R) physical bins via physical_route_fn."""
    num_ep, local_experts, num_replicas = 2, 2, 1
    num_slots = local_experts + num_replicas
    total_bins = num_ep * num_slots  # 6 physical bins: [0,1,2] on shard 0; [3,4,5] on shard 1.
    layer = self._make_layer(num_ep=num_ep, num_experts=num_ep * local_experts, num_experts_per_tok=2)

    inputs = jnp.arange(2 * 3 * 16, dtype=jnp.float32).reshape(2, 3, 16)
    gate_logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4), dtype=jnp.float32)
    # 6 tokens * top-2 = 12 selections; expert 0 is selected 6 times.
    forced = jnp.array(
        [
            [[0, 1], [0, 2], [0, 3]],
            [[0, 1], [0, 2], [0, 3]],
        ],
        dtype=jnp.int32,
    )
    # Map logical [0, 1, 2, 3] -> home bins [0, 1, 3, 4], rerouting the second half of expert-0 hits to replica bin 5.
    base_map = jnp.array([0, 1, 3, 4], dtype=jnp.int32)
    replica_bin = 5

    def physical_route_fn(flat_logical):
      phys = base_map[flat_logical]
      positions = jnp.arange(flat_logical.shape[0])
      reroute = (flat_logical == 0) & (positions >= 6)
      return jnp.where(reroute, replica_bin, phys), total_bins

    sorted_inputs, _, _, group_sizes, *_ = layer.permute(
        inputs,
        gate_logits,
        None,
        use_custom_sort_vjp=False,
        rngs=jax.random.PRNGKey(1),
        forced_routed_experts=forced,
        compute_lb_loss=False,
        physical_route_fn=physical_route_fn,
    )
    self.assertEqual(group_sizes.shape, (total_bins,))
    self.assertEqual(int(jnp.sum(group_sizes)), forced.size)
    # Bins: [home0=3, home1=2, rep_shard0=0, home2=2, home3=2, rep_shard1=3].
    np.testing.assert_array_equal(np.asarray(group_sizes), np.array([3, 2, 0, 2, 2, 3], dtype=np.int32))
    # Last 3 rows of sorted_inputs belong to replica_bin=5, which came from batch index 1 tokens (positions 6, 8, 10).
    np.testing.assert_allclose(np.asarray(sorted_inputs[-3:]), np.asarray(inputs[1]), atol=1e-6)

  def test_lagged_plan_updates_batch_stat(self):
    """Verifies lagged mode updates expert_replica_load BatchStat with zero gradient leakage."""
    num_ep, num_experts = 2, 4
    layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=1,
        expert_replica_plan="lagged",
    )
    self.assertIsInstance(layer.expert_replica_load, nnx.BatchStat)
    np.testing.assert_array_equal(np.asarray(layer.expert_replica_load[...]), np.zeros((num_experts,), np.float32))

    rng = jax.random.PRNGKey(7)
    k_in, k_log, k_w0, k_w1, k_wo = jax.random.split(rng, 5)
    inputs = jax.random.normal(k_in, (4, 4, 16), jnp.float32)
    logits = jax.random.normal(k_log, (4, 4, num_experts), jnp.float32)
    w0 = jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.2
    w1 = jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.2
    wo = jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.2
    forced = jnp.array(
        [
            [[0, 1], [0, 1], [0, 2], [0, 1]],
            [[0, 1], [0, 1], [0, 3], [0, 1]],
            [[0, 1], [0, 2], [0, 1], [0, 1]],
            [[0, 1], [0, 3], [2, 3], [0, 1]],
        ],
        dtype=jnp.int32,
    )

    with self._cpu_ep_runtime(layer, num_ep=num_ep):
      # Differentiate the recorded BatchStat sum wrt inputs and gate_logits (using top-k routing) to verify
      # jax.lax.stop_gradient prevents any gradient leakage through expert_replica_load.
      def stat_only_loss(m, x, g):
        _ = m.sparse_matmul(x, g, None, w0, w1, wo, None, None, None)
        return jnp.sum(m.expert_replica_load[...])

      gx, gg = nnx.grad(stat_only_loss, argnums=(1, 2))(layer, inputs, logits)
      np.testing.assert_array_equal(np.asarray(gx), np.zeros_like(inputs))
      np.testing.assert_array_equal(np.asarray(gg), np.zeros_like(logits))

      # Reset to zero, then run step 1 with forced_routed_experts and check exact per-expert counts.
      layer.expert_replica_load[...] = jnp.zeros((num_experts,), jnp.float32)
      _ = layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)
      expected_counts = np.bincount(np.asarray(forced).ravel(), minlength=num_experts).astype(np.float32)
      np.testing.assert_array_equal(np.asarray(layer.expert_replica_load.value), expected_counts)

    # Step 2 consumes the non-zero lagged_load recorded in step 1, places a replica on shard 1, and matches R=0.
    cotangent = jax.random.normal(jax.random.PRNGKey(99), (4, 4, 16), jnp.float32)
    captured_step2 = {}
    with self._cpu_ep_runtime(layer, num_ep=num_ep, cotangent=cotangent, captured=captured_step2):
      out_lagged, _, _ = layer.sparse_matmul(
          inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced
      )
    self.assertGreater(int(np.asarray(captured_step2["local_group_sizes"])[1, -1]), 0)

    base_layer = self._make_layer(num_ep=num_ep, num_experts=num_experts, num_experts_per_tok=2)
    captured_base = {}
    with self._cpu_ep_runtime(base_layer, num_ep=num_ep, cotangent=cotangent, captured=captured_base):
      out_base, _, _ = base_layer.sparse_matmul(
          inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced
      )
    np.testing.assert_allclose(np.asarray(out_lagged), np.asarray(out_base), rtol=1e-5, atol=1e-5)
    for g_lag, g_ref in zip(captured_step2["grads"], captured_base["grads"]):
      np.testing.assert_allclose(np.asarray(g_lag), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("concat_direct", "concat", "direct", "rounds", False, False),
      ("concat_gather", "concat", "gather", "greedy", False, False),
      ("split", "split", "direct", "rounds", False, False),
      ("overhead_control", "concat", "direct", "rounds", True, False),
      ("split_return_weights", "split", "direct", "rounds", False, True),
  )
  def test_end_to_end_sparse_matmul_parity_on_cpu(self, impl, move, planner, overhead_control, return_weights):
    """Verifies forward output and (dx, dlogits, dw0, dw1, dwo) match R=0 across all replication modes."""
    num_ep, num_experts = 2, 4
    rng = jax.random.PRNGKey(42)
    k_in, k_log, k_w0, k_w1, k_wo, k_cot = jax.random.split(rng, 6)
    inputs = jax.random.normal(k_in, (4, 4, 16), jnp.float32)
    logits = jax.random.normal(k_log, (4, 4, num_experts), jnp.float32)
    w0 = jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.25
    w1 = jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.25
    wo = jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.25
    cotangent = jax.random.normal(k_cot, (4, 4, 16), jnp.float32)
    # Skew traffic toward expert 0 on shard 0 while keeping shard 1 (experts 2, 3) cold so a replica is placed.
    forced = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [0, 3]],
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )

    def run_fwd_bwd(layer, force_ret_w=False):
      captured = {}
      with self._cpu_ep_runtime(
          layer, num_ep=num_ep, cotangent=cotangent, force_return_weights=force_ret_w, captured=captured
      ):
        out, _, _ = layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)
      return out, captured["grads"], captured

    base_layer = self._make_layer(
        num_ep=num_ep, num_experts=num_experts, num_experts_per_tok=2, expert_replicas_per_device=0
    )
    out_ref, grads_ref, _ = run_fwd_bwd(base_layer)

    rep_layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=1,
        expert_replica_impl=impl,
        expert_replica_move=move,
        expert_replica_planner=planner,
        expert_replica_overhead_control=overhead_control,
    )
    out_rep, grads_rep, captured = run_fwd_bwd(rep_layer, force_ret_w=return_weights)

    # Confirm the replica slot on shard 1 actually received tokens (unless overhead_control zeroes replica routing).
    local_gs = np.asarray(captured["local_group_sizes"])  # shape: [num_ep, local_experts + num_replicas]
    if overhead_control:
      self.assertEqual(int(local_gs[:, -1].sum()), 0)
    else:
      self.assertGreater(int(local_gs[1, -1]), 0)

    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for name, g_rep, g_ref in zip(("dx", "dlogits", "dw0", "dw1", "dwo"), grads_rep, grads_ref):
      with self.subTest(grad=name):
        np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

    if return_weights:
      ret_w0, ret_w1, ret_wo = captured["ret_weights"]
      np.testing.assert_allclose(np.asarray(ret_w0).reshape(w0.shape), np.asarray(w0), atol=1e-6)
      np.testing.assert_allclose(np.asarray(ret_w1).reshape(w1.shape), np.asarray(w1), atol=1e-6)
      np.testing.assert_allclose(np.asarray(ret_wo).reshape(wo.shape), np.asarray(wo), atol=1e-6)

  def test_guardrails_raise_on_unsupported_configs(self):
    """Verifies unsupported MoE configurations raise explicit errors when expert replication is enabled."""
    inputs = jnp.ones((4, 4, 16), jnp.float32)
    logits = jnp.ones((4, 4, 4), jnp.float32)
    w0 = jnp.ones((4, 16, 8), jnp.float32)
    w1 = jnp.ones((4, 16, 8), jnp.float32)
    wo = jnp.ones((4, 8, 16), jnp.float32)

    with self.subTest("use_ring_of_experts"):
      layer = self._make_layer(expert_replicas_per_device=1)
      object.__setattr__(layer.config, "use_ring_of_experts", True)
      with self._cpu_ep_runtime(layer, num_ep=2):
        with self.assertRaisesRegex(NotImplementedError, "ragged all-to-all path"):
          layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

    with self.subTest("num_moe_emb_chunks"):
      layer = self._make_layer(expert_replicas_per_device=1)
      object.__setattr__(layer.config, "num_moe_emb_chunks", 2)
      with self._cpu_ep_runtime(layer, num_ep=2):
        with self.assertRaisesRegex(NotImplementedError, "num_moe_emb_chunks"):
          layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

    with self.subTest("num_experts_not_divisible_by_num_ep"):
      layer = self._make_layer(num_ep=2, num_experts=4, expert_replicas_per_device=1)
      object.__setattr__(layer.config, "num_experts", 5)
      with self._cpu_ep_runtime(layer, num_ep=2):
        with self.assertRaisesRegex(ValueError, "divisible by num_ep"):
          layer.sparse_matmul(inputs, jnp.ones((4, 4, 5), jnp.float32), None, w0, w1, wo, None, None, None)

    with self.subTest("moe_bias"):
      layer = self._make_layer(expert_replicas_per_device=1)
      w0_bias = jnp.zeros((4, 8), jnp.float32)
      with self._cpu_ep_runtime(layer, num_ep=2):
        with self.assertRaisesRegex(NotImplementedError, "MoE biases"):
          layer.sparse_matmul(inputs, logits, None, w0, w1, wo, w0_bias, None, None)


if __name__ == "__main__":
  absltest.main()

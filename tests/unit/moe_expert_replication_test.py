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
    """Initializes a Mixtral EP=2 config on a v5e-8 compile topology."""
    base_kw = {
        "run_name": "moe_expert_replication_test",
        "base_output_directory": "/tmp/moe_expert_replication_test",
        "enable_checkpointing": False,
        "compile_topology": "v5e-8",
        "compile_topology_num_slices": 1,
        "ici_expert_parallelism": 2,
        "ici_fsdp_parallelism": 4,
        "model_name": "mixtral-8x7b",
        "override_model_config": True,
        "sparse_matmul": True,
        "megablox": True,
        "use_ring_of_experts": False,
    }
    base_kw.update(kw)
    return pyconfig.initialize([None, get_test_config_path()], **base_kw)

  def test_defaults_off(self):
    cfg = self._init()
    self.assertEqual(cfg.expert_replicas_per_device, 0)
    self.assertEqual(cfg.expert_replica_max_send, 4)
    self.assertEqual(cfg.expert_replica_board_size, 8)

  def test_flags_parse(self):
    cfg = self._init(
        expert_replicas_per_device=4,
        expert_replica_max_send=3,
        expert_replica_board_size=16,
    )
    self.assertEqual(cfg.expert_replicas_per_device, 4)
    self.assertEqual(cfg.expert_replica_max_send, 3)
    self.assertEqual(cfg.expert_replica_board_size, 16)

  def test_invalid_bounds_rejected(self):
    for kw in (
        {"expert_replicas_per_device": -1},
        {"expert_replica_max_send": 0},
        {"expert_replica_max_send": 9},
        {"expert_replica_board_size": 0},
    ):
      with self.subTest(kw=kw):
        with self.assertRaises((pydantic.ValidationError, ValueError)):
          self._init(**kw)

  def test_expert_replica_max_send_capped_at_8(self):
    cfg_ok = self._init(expert_replica_max_send=8)
    self.assertEqual(cfg_ok.expert_replica_max_send, 8)
    with self.assertRaises((pydantic.ValidationError, ValueError)):
      self._init(expert_replica_max_send=9)

  def test_config_validator_rejects_unsupported_combinations(self):
    """Verifies MaxTextConfig fails before compile on unsupported flag combinations."""
    bad_cases = [
        ("ring_of_experts", {"use_ring_of_experts": True}, "use_ring_of_experts=False"),
        ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks > 0"),
        ("mlp_bias", {"mlp_bias": True}, "mlp_bias=True"),
        (
            "chunk_wgrad",
            {"moe_accumulate_chunk_wgrad": True},
            "moe_accumulate_chunk_wgrad=True",
        ),
        (
            "dropless_fallback",
            {"moe_dropless_fallback": "layer", "use_ragged_sort": True, "ragged_buffer_factor": 1.25},
            "moe_dropless_fallback",
        ),
        ("ep_1", {"ici_expert_parallelism": 1, "ici_fsdp_parallelism": 8}, "expert parallelism > 1"),
        ("no_sparse_matmul", {"sparse_matmul": False}, "sparse_matmul=True"),
        ("num_experts_1", {"num_experts": 1, "num_experts_per_tok": 1}, "num_experts > 1"),
        ("fp8_moe", {"fp8_moe": True}, "FP8 or quantized MoE"),
        ("fp8_moe_fake_quant", {"fp8_moe_fake_quant": True}, "FP8 or quantized MoE"),
        ("quantization", {"quantization": "int8"}, "FP8 or quantized MoE"),
        ("te_gmm_quantization", {"te_gmm_quantization": "te_mxfp8"}, "FP8 or quantized MoE"),
    ]
    for label, overrides, expected_msg in bad_cases:
      with self.subTest(case=label):
        with self.assertRaisesRegex(ValueError, expected_msg):
          self._init(expert_replicas_per_device=2, **overrides)


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
  def _cpu_ep_runtime(
      self, layer, num_ep=2, cotangent=None, force_return_weights=False, captured=None, wrap_shard_fn=None
  ):
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
          outs = wrap_shard_fn(fn, s_args) if wrap_shard_fn is not None else fn(*s_args)
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

  def test_permute_with_physical_route_fn(self):
    """Verifies permute() remaps logical experts into EP*(L+R) physical bins via physical_route_fn."""
    num_ep, local_experts, num_replicas = 2, 2, 1
    num_slots = local_experts + num_replicas
    total_bins = num_ep * num_slots
    layer = self._make_layer(num_ep=num_ep, num_experts=num_ep * local_experts, num_experts_per_tok=2)

    inputs = jnp.arange(2 * 3 * 16, dtype=jnp.float32).reshape(2, 3, 16)
    gate_logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4), dtype=jnp.float32)
    forced = jnp.array(
        [
            [[0, 1], [0, 2], [0, 3]],
            [[0, 1], [0, 2], [0, 3]],
        ],
        dtype=jnp.int32,
    )
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
    np.testing.assert_array_equal(np.asarray(group_sizes), np.array([3, 2, 0, 2, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(sorted_inputs[-3:]), np.asarray(inputs[1]), atol=1e-6)

  def test_end_to_end_sparse_matmul_parity_and_required_rbf(self):
    """Verifies forward output, (dx, dlogits, dw0, dw1, dwo), and lowered required_rbf match R=0 on skewed traffic."""
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

    def run_fwd_bwd(layer):
      captured = {}
      with self._cpu_ep_runtime(layer, num_ep=num_ep, cotangent=cotangent, captured=captured):
        out, _, _ = layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced)
      req = float(np.asarray(nnx.pop(layer, nnx.Intermediate)["moe_required_rbf"].get_value()[0]))
      return out, captured["grads"], captured, req

    base_layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=0,
        log_required_ragged_buffer_factor=True,
    )
    out_ref, grads_ref, _, req_base = run_fwd_bwd(base_layer)

    rep_layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        expert_replica_board_size=2,
        log_required_ragged_buffer_factor=True,
    )
    out_rep, grads_rep, captured, req_rep = run_fwd_bwd(rep_layer)

    # Replica slot on shard 1 received offloaded expert-0 tokens and reduced required_rbf.
    local_gs = np.asarray(captured["local_group_sizes"])
    self.assertGreater(int(local_gs[1, -1]), 0)
    self.assertLess(req_rep, req_base)

    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for name, g_rep, g_ref in zip(("dx", "dlogits", "dw0", "dw1", "dwo"), grads_rep, grads_ref):
      with self.subTest(grad=name):
        np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

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

    with self.subTest("return_weights"):
      layer = self._make_layer(expert_replicas_per_device=1)
      with self._cpu_ep_runtime(layer, num_ep=2, force_return_weights=True):
        with self.assertRaisesRegex(NotImplementedError, "moe_accumulate_chunk_wgrad"):
          layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

    with self.subTest("ep_1_runtime"):
      layer = self._make_layer(num_ep=1, expert_replicas_per_device=1)
      with self._cpu_ep_runtime(layer, num_ep=1):
        with self.assertRaisesRegex(ValueError, "requires num_ep > 1"):
          layer.sparse_matmul(inputs, logits, None, w0, w1, wo, None, None, None)

    with self.subTest("unsharded_decode_batch_size_1"):
      layer = self._make_layer(num_ep=2, expert_replicas_per_device=1)

      def _replicated_shard_map(fn, **_kw):
        def _run(*args):
          return jax.vmap(lambda _i: fn(*args), axis_name="expert")(jnp.arange(2))

        return _run

      with mock.patch.object(jax, "shard_map", side_effect=_replicated_shard_map):
        with self.assertRaisesRegex(ValueError, "is_batch_sharded_by_expert=True"):
          layer.sparse_matmul(inputs[:1], logits[:1], None, w0[:2], w1[:2], wo[:2], None, None, None)

  def test_route_metadata_with_replica_plan_is_valid_pytree(self):
    """RouteMetadata carrying a ReplicaPlan flattens as a PyTree and crosses jax.jit and jax.lax.cond."""
    from maxtext.layers import moe_replica_planner as mrp  # pylint: disable=import-outside-toplevel

    counts = jnp.asarray([[20, 2, 1, 1], [10, 1, 1, 1]], dtype=jnp.int32)
    plan = mrp.compute_plan_iter(counts, num_ep=2, num_replicas=1, max_send=1, board_size=1)
    rm = moe.RouteMetadata(
        expert_shard_id=0,
        local_sorted_indices=jnp.arange(4, dtype=jnp.int32),
        reshaped_group_sizes=jnp.asarray([12, 12], dtype=jnp.int32),
        all_shards_group_sizes=jnp.ones((2, 2), dtype=jnp.int32),
        replica_plan=plan,
    )
    leaves, treedef = jax.tree_util.tree_flatten(rm)
    self.assertLen(leaves, 4 + 5)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(restored.replica_plan.num_slots, plan.num_slots)

    @jax.jit
    def roundtrip(flag, meta):
      return jax.lax.cond(flag, lambda m: m, lambda m: m, meta)

    out = roundtrip(jnp.asarray(True), rm)
    np.testing.assert_array_equal(np.asarray(out.replica_plan.route_table), np.asarray(plan.route_table))

  def test_routed_bias_updates_match_r0(self):
    """Verifies routed_bias_update_rate > 0 emits logical expert bias updates identical to R=0."""
    num_ep, num_experts = 2, 4
    rng = jax.random.PRNGKey(7)
    k_in, k_log, k_w0, k_w1, k_wo, k_cot = jax.random.split(rng, 6)
    inputs = jax.random.normal(k_in, (4, 4, 16), jnp.float32)
    # Skew logits strongly toward expert 0 on shard 0 so replication offloads tokens to shard 1.
    logits = jax.random.normal(k_log, (4, 4, num_experts), jnp.float32).at[:, :, 0].add(8.0).at[:, :, 1].add(7.0)
    pre_bias_logits = logits
    w0 = jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.25
    w1 = jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.25
    wo = jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.25
    cotangent = jax.random.normal(k_cot, (4, 4, 16), jnp.float32)

    deepseek_kw = {
        "decoder_block": "deepseek",
        "routed_score_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "n_routing_groups": -1,
        "topk_routing_group": -1,
        "routed_bias": True,
        "routed_bias_update_rate": 1e-3,
    }
    base_layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=0,
        **deepseek_kw,
    )
    cap_base = {}
    with self._cpu_ep_runtime(base_layer, num_ep=num_ep, cotangent=cotangent, captured=cap_base):
      out_ref, _, bias_ref = base_layer.sparse_matmul(inputs, logits, pre_bias_logits, w0, w1, wo, None, None, None)

    rep_layer = self._make_layer(
        num_ep=num_ep,
        num_experts=num_experts,
        num_experts_per_tok=2,
        expert_replicas_per_device=1,
        expert_replica_max_send=2,
        expert_replica_board_size=2,
        **deepseek_kw,
    )
    cap_rep = {}
    with self._cpu_ep_runtime(rep_layer, num_ep=num_ep, cotangent=cotangent, captured=cap_rep):
      out_rep, _, bias_rep = rep_layer.sparse_matmul(inputs, logits, pre_bias_logits, w0, w1, wo, None, None, None)

    self.assertIsNotNone(bias_ref)
    self.assertIsNotNone(bias_rep)
    self.assertGreater(int(np.asarray(cap_rep["local_group_sizes"])[1, -1]), 0)
    np.testing.assert_allclose(np.asarray(bias_rep), np.asarray(bias_ref), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for g_rep, g_ref in zip(cap_rep["grads"], cap_base["grads"]):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

  def test_remat_and_multi_branch_trace_parity_with_r0(self):
    """Verifies explicit RouteMetadata.replica_plan threading under jax.checkpoint remat and lax.cond."""
    num_ep, num_experts = 2, 4
    rng = jax.random.PRNGKey(19)
    k_in, k_log, k_w0, k_w1, k_wo, k_cot = jax.random.split(rng, 6)
    inputs = jax.random.normal(k_in, (4, 4, 16), jnp.float32)
    logits = jax.random.normal(k_log, (4, 4, num_experts), jnp.float32)
    w0 = jax.random.normal(k_w0, (num_experts, 16, 8), jnp.float32) * 0.25
    w1 = jax.random.normal(k_w1, (num_experts, 16, 8), jnp.float32) * 0.25
    wo = jax.random.normal(k_wo, (num_experts, 8, 16), jnp.float32) * 0.25
    cotangent = jax.random.normal(k_cot, (4, 4, 16), jnp.float32)
    forced_a = jnp.array(
        [
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [0, 3]],
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            [[0, 1], [0, 1], [0, 1], [2, 3]],
        ],
        dtype=jnp.int32,
    )

    alt_forced = jnp.array(
        [
            [[2, 3], [2, 3], [2, 3], [0, 2]],
            [[2, 3], [2, 3], [2, 3], [1, 3]],
        ],
        dtype=jnp.int32,
    )

    def wrap_remat_and_cond(fn, s_args):
      @jax.checkpoint
      def ckpt_shard(x_, g_, w0_, w1_, wo_, forced_main):
        def branch_true(_):
          return fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], forced_main, s_args[12])

        def branch_false(_):
          return fn(x_, g_, s_args[2], w0_, w1_, wo_, *s_args[6:11], alt_forced, s_args[12])

        return jax.lax.cond(jnp.bool_(True), branch_true, branch_false, operand=None)

      return ckpt_shard(s_args[0], s_args[1], s_args[3], s_args[4], s_args[5], s_args[11])

    def eval_remat_and_cond(replicas):
      layer = self._make_layer(
          num_ep=num_ep,
          num_experts=num_experts,
          num_experts_per_tok=2,
          expert_replicas_per_device=replicas,
          expert_replica_max_send=2,
          expert_replica_board_size=2,
      )
      cap = {}
      with self._cpu_ep_runtime(
          layer, num_ep=num_ep, cotangent=cotangent, captured=cap, wrap_shard_fn=wrap_remat_and_cond
      ):
        out, _, _ = layer.sparse_matmul(
            inputs, logits, None, w0, w1, wo, None, None, None, forced_routed_experts=forced_a
        )
      return out, cap["grads"]

    out_ref, grads_ref = eval_remat_and_cond(0)
    out_rep, grads_rep = eval_remat_and_cond(1)
    np.testing.assert_allclose(np.asarray(out_rep), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for g_rep, g_ref in zip(grads_rep, grads_ref):
      np.testing.assert_allclose(np.asarray(g_rep), np.asarray(g_ref), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()

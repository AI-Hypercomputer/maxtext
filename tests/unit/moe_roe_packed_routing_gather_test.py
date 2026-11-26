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
"""Unit tests for `moe_roe_packed_routing_gather` and `RematAndOffload.moe_routing_maps`."""

import contextlib
import os
import re
import sys
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import yaml

from maxtext.configs import pyconfig
from maxtext.configs import types as mt_types
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import globals as mt_globals
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_post_train_test_config_path, get_test_config_path

_NUM_EXPERTS = 16
_TOP_K = 4
_EMB = 128
_SEQ = 32
_REL_TOL = 1e-5
_BF16_REL_NORM_TOL = 6e-2


def _cfg(**overrides):
  """Shrunk Qwen3.5 Ring-of-Experts config running on up to 4 host/TPU devices."""
  num_devices = jax.device_count()
  cp = 2 if num_devices >= 4 else 1
  ep = 2 if num_devices >= 2 else 1
  fsdp = max(1, num_devices // (cp * ep))
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
      "base_mlp_dim": 128,
      "base_moe_mlp_dim": 128,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": 1.0,
      "dtype": "bfloat16",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "float32_gate_logits": True,
      # Off TPU, megablox runs its Pallas GMM kernel in interpret mode; sparse_matmul=True keeps the routed
      # Ring-of-Experts path via jax.lax.ragged_dot.
      "megablox": False,
      "sparse_matmul": True,
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_gather_fallback": True,
      "ragged_gather_reduce_fallback": True,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": fsdp,
      "ici_context_parallelism": cp,
      "ici_expert_parallelism": ep,
      "context_parallel_load_balance": False,
      "moe_topk_before_ep_all_gather": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class PackRoutingHelperTest(parameterized.TestCase):
  """`pack_routing_for_all_gather` / `unpack_routing` exact bit-level round-trip."""

  def test_tier2_default_false_and_not_in_base_yml(self):
    field = mt_types.MoEGeneral.model_fields["moe_roe_packed_routing_gather"]
    self.assertIs(field.annotation, bool)
    self.assertIs(field.default, False)
    base_yml = os.path.join(mt_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
    with open(base_yml, encoding="utf-8") as f:
      self.assertNotIn("moe_roe_packed_routing_gather", yaml.safe_load(f))

  @parameterized.named_parameters(("float32_weights", jnp.float32), ("bfloat16_weights", jnp.bfloat16))
  def test_round_trip_is_exact_above_bf16_integer_limit(self, weights_dtype):
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    weights = jax.random.uniform(k0, (2, 64, 10), jnp.float32).astype(weights_dtype)
    # Includes -1 and odd expert ids up to 4095 (> 256, where bfloat16 loses integer precision).
    ids = (jax.random.permutation(k1, 2 * 64 * 10) % 4097 - 1).reshape(2, 64, 10).astype(jnp.int32)
    packed = moe.pack_routing_for_all_gather(weights, ids)
    self.assertEqual(packed.dtype, jnp.float32)
    self.assertEqual(packed.shape, (2, 64, 20))
    got_weights, got_ids = moe.unpack_routing(packed, weights.dtype, ids.dtype)
    self.assertEqual((got_weights.dtype, got_ids.dtype), (weights.dtype, ids.dtype))
    np.testing.assert_array_equal(np.asarray(got_ids), np.asarray(ids))
    np.testing.assert_array_equal(
        np.ascontiguousarray(got_weights).view(np.uint8),
        np.ascontiguousarray(weights).view(np.uint8),
    )


class PackedRoutingGatherValidatorTest(parameterized.TestCase):
  """Config validation rejects invalid flag combinations cleanly."""

  @parameterized.named_parameters(
      (
          "dense",
          {
              "sparse_matmul": False,
              "use_ring_of_experts": False,
              "use_ragged_sort": False,
              "ragged_buffer_factor": -1.0,
              "moe_topk_before_ep_all_gather": False,
          },
          "sparse_matmul=True",
      ),
      (
          "te_moe_block",
          {
              "te_moe_block": True,
              "use_ring_of_experts": False,
              "use_ragged_sort": False,
              "moe_topk_before_ep_all_gather": False,
          },
          "te_moe_block=False",
      ),
      (
          "no_roe",
          {"use_ring_of_experts": False, "use_ragged_sort": False, "moe_topk_before_ep_all_gather": False},
          "use_ring_of_experts=True",
      ),
      ("no_topk_before_ag", {"moe_topk_before_ep_all_gather": False}, "moe_topk_before_ep_all_gather=True"),
      ("random_routing", {"use_random_routing": True}, "use_random_routing=False"),
      ("emb_chunks", {"num_moe_emb_chunks": 2}, "num_moe_emb_chunks <= 0"),
  )
  def test_rejected(self, overrides, requirement):
    if jax.device_count() < 2:
      self.skipTest("Validator test uses EP>=2 base _cfg.")
    with self.assertRaisesRegex(ValueError, f"moe_roe_packed_routing_gather=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_roe_packed_routing_gather=True, **overrides)

  def test_qa2a_rowwise_knobs_rejected_in_pretrain_and_rl_and_ring_combine_bwd_accepted(self):
    """EP ragged-a2a FP8 knobs reject Ring-of-Experts packed routing gather in both MaxTextConfig and RLConfig."""
    common = {
        "model_name": "deepseek3-test",
        "override_model_config": True,
        "enable_checkpointing": False,
        "skip_jax_distributed_system": True,
        "compile_topology": "tpu7x-8",
        "compile_topology_num_slices": 1,
        "ici_expert_parallelism": 2,
        "use_ring_of_experts": True,
        "moe_topk_before_ep_all_gather": True,
        "moe_roe_packed_routing_gather": True,
    }
    pretrain_argv = [
        os.path.join(mt_globals.MAXTEXT_PKG_DIR, "trainers", "pre_train", "train.py"),
        get_test_config_path(),
    ]
    rl_argv = [
        os.path.join(mt_globals.MAXTEXT_PKG_DIR, "trainers", "post_train", "rl", "train_rl.py"),
        get_post_train_test_config_path(),
    ]
    for knob in (
        "moe_quantize_dispatch_fwd_method",
        "moe_quantize_combine_fwd_method",
        "moe_quantize_dispatch_bwd_method",
    ):
      for argv in (pretrain_argv, rl_argv):
        with self.assertRaisesRegex(
            ValueError, "quantizes the EP ragged all-to-all and requires use_ring_of_experts=False"
        ):
          pyconfig.initialize(argv, run_name="packed_qa2a_reject", **common, **{knob: "rowwise"})
    for argv in (pretrain_argv, rl_argv):
      with self.assertRaisesRegex(ValueError, "with use_ring_of_experts=True"):
        pyconfig.initialize(
            argv, run_name="packed_qa2a_cbwd_reject", **common, moe_quantize_combine_bwd_method="rowwise@fp4"
        )
      for valid_method in ("rowwise", "rowwise@e4m3"):
        cfg_ok = pyconfig.initialize(
            argv, run_name="packed_ring_cbwd_ok", **common, moe_quantize_combine_bwd_method=valid_method
        )
        self.assertEqual(cfg_ok.moe_quantize_combine_bwd_method, valid_method)
      with self.assertRaisesRegex(
          ValueError, "moe_roe_packed_routing_gather=True requires: .*moe_topk_before_ep_all_gather=True"
      ):
        pyconfig.initialize(argv, run_name="packed_rl_reject", **{**common, "moe_topk_before_ep_all_gather": False})


def _eqns(jaxpr):
  for eqn in jaxpr.eqns:
    yield eqn
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
        if hasattr(sub, "eqns"):
          yield from _eqns(sub)


def _structure(jaxpr, seq_per_cp):
  """Extracts all-gather and checkpoint-name counts from a traced forward JAXpr."""
  eqns = list(_eqns(jaxpr))
  gathers = [tuple(v.aval.shape) for e in eqns if e.primitive.name == "all_gather" for v in e.outvars]
  bool_gathers = [
      tuple(v.aval.shape)
      for e in eqns
      if e.primitive.name == "all_gather"
      for v in e.outvars
      if v.aval.dtype == jnp.bool_
  ]
  names = [
      (e.params.get("name"), tuple(e.invars[0].aval.shape), e.invars[0].aval.dtype)
      for e in eqns
      if e.primitive.name == "name"
  ]
  return {
      "topk_gathers": sum(len(s) == 3 and s[-1] == _TOP_K for s in gathers),
      "packed_gathers": sum(len(s) == 3 and s[1:] == (seq_per_cp, 2 * _TOP_K) for s in gathers),
      "packed_tags": sorted({n for n, s, _ in names if len(s) == 3 and s[1:] == (seq_per_cp, 2 * _TOP_K)}),
      "mask_gathers": sum(len(s) == 2 and s[1] == seq_per_cp for s in bool_gathers),
      "mask_tags": sorted({n for n, s, dt in names if len(s) == 2 and s[1] == seq_per_cp and dt == jnp.bool_}),
      "legacy_moe_route_tags": sum(n == "moe_route" for n, _, _ in names),
  }


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
  """Constructs synthetic hidden states, optional forced_routed_experts, and token_mask."""
  batch = int(cfg.per_device_batch_size * jax.device_count())
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
    lens = jnp.array([max(4, _SEQ - 4 * (b + 1)) for b in range(batch)])
    token_mask = jnp.arange(_SEQ)[None, :] < lens[:, None]
  return x, forced, token_mask


class PackedRoutingGatherParityAndRematTest(parameterized.TestCase):
  """Multi-device parity, HLO gather counts, moe_routing_maps remat/offload, and negative controls."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest("Requires >= 2 devices for Ring-of-Experts EP all-gather verification.")

  def _run(self, cfg, mesh, params, x, forced, token_mask, use_remat=False):
    """Runs forward + backward value_and_grad and returns outputs, jaxpr structure, and HLO."""
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

      def forward_layer(p, inp):
        return nnx.merge(graphdef, p, rest)(inp, forced_routed_experts=forced, token_mask=token_mask)

      if use_remat:
        save_names = ["moe_routing_maps"] if cfg.moe_routing_maps == "device" else []
        offload_names = ["moe_routing_maps"] if cfg.moe_routing_maps == "offload" else []
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=save_names,
            names_which_can_be_offloaded=offload_names,
            offload_src="device",
            offload_dst="pinned_host",
        )
        forward_layer = jax.checkpoint(forward_layer, policy=policy)

      def loss_fn(p, inp):
        out, lb_loss, bias_upd = forward_layer(p, inp)
        loss = jnp.mean(out.astype(jnp.float32) ** 2)
        if lb_loss is not None:
          loss = loss + lb_loss.astype(jnp.float32)
        return loss, (out, bias_upd)

      seq_per_cp = _SEQ // cfg.ici_context_parallelism
      structure = _structure(jax.jit(loss_fn).trace(params, x).jaxpr.jaxpr, seq_per_cp)
      grad_raw = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
      grad_fn = jax.jit(grad_raw)
      grad_traced = grad_fn.trace(params, x)
      grad_eqns = list(_eqns(grad_traced.jaxpr.jaxpr))
      structure["grad_packed_gathers"] = sum(
          len(v.aval.shape) == 3 and v.aval.shape[1:] == (seq_per_cp, 2 * _TOP_K)
          for e in grad_eqns
          if e.primitive.name == "all_gather"
          for v in e.outvars
      )
      structure["packed_host_puts"] = sum(
          "Host" in str(e.params.get("devices"))
          and any(len(v.aval.shape) == 3 and v.aval.shape[-1] == 2 * _TOP_K for v in e.invars)
          for e in grad_eqns
          if e.primitive.name == "device_put"
      )
      structure["packed_device_puts"] = sum(
          "Device" in str(e.params.get("devices"))
          and any(len(v.aval.shape) == 3 and v.aval.shape[-1] == 2 * _TOP_K for v in e.invars)
          for e in grad_eqns
          if e.primitive.name == "device_put"
      )
      structure["grad_mask_gathers"] = sum(
          len(v.aval.shape) == 2 and v.aval.shape[1] == seq_per_cp and v.aval.dtype == jnp.bool_
          for e in grad_eqns
          if e.primitive.name == "all_gather"
          for v in e.outvars
      )
      structure["mask_host_puts"] = sum(
          "Host" in str(e.params.get("devices"))
          and any(len(v.aval.shape) == 2 and v.aval.dtype == jnp.bool_ for v in e.invars)
          for e in grad_eqns
          if e.primitive.name == "device_put"
      )
      structure["mask_device_puts"] = sum(
          "Device" in str(e.params.get("devices"))
          and any(len(v.aval.shape) == 2 and v.aval.dtype == jnp.bool_ for v in e.invars)
          for e in grad_eqns
          if e.primitive.name == "device_put"
      )
      # Compile once from the traced grad_fn and reuse the executable for both HLO inspection and execution.
      compiled = grad_traced.lower().compile()
      if not (use_remat and cfg.moe_routing_maps == "offload"):
        compiled_hlo = compiled.as_text() or ""
        structure["compiled_bool_all_gathers"] = sum(
            bool(re.search(r"=\s*(?:pred\[\S*|\([^)]*\bpred\[[^)]*\))\s+all-gather(?:-start)?\(", line))
            for line in compiled_hlo.splitlines()
        )
      (loss, (out, bias_upd)), grads = compiled(params, x)
    return jax.device_get((loss, out, bias_upd, grads)), structure

  def _assert_case(self, overrides, mode="real", compare="bitwise", use_remat=False, control=None):
    """Compares unpacked vs packed routing gather structure and numerical outputs/gradients."""
    cfg_old = _cfg(**overrides)
    cfg_new = _cfg(**overrides, moe_roe_packed_routing_gather=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
    x, forced, token_mask = _inputs(cfg_old, mode)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
      _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)

    (loss_old, out_old, bias_old, grads_old), s_old = self._run(
        cfg_old, mesh, params, x, forced, token_mask, use_remat=use_remat
    )
    with control if control is not None else contextlib.nullcontext():
      (loss_new, out_new, bias_new, grads_new), s_new = self._run(
          cfg_new, mesh, params, x, forced, token_mask, use_remat=use_remat
      )

    n_chunks = cfg_new.num_moe_token_chunks
    want_tags = ["moe_routing_maps"] if cfg_new.moe_routing_maps != "remat" else []
    structure_ok = (
        s_old["topk_gathers"] == 2 * n_chunks
        and s_old["packed_gathers"] == 0
        and s_new["topk_gathers"] == 0
        and s_new["packed_gathers"] == 1
        and s_new["packed_tags"] == want_tags
        and s_old["legacy_moe_route_tags"] == s_new["legacy_moe_route_tags"] == 0
    )

    named = [("output", out_old, out_new), ("loss", loss_old, loss_new)] + [
        (jax.tree_util.keystr(path), a, b)
        for (path, a), b in zip(
            jax.tree_util.tree_flatten_with_path(grads_old)[0],
            jax.tree_util.tree_leaves(grads_new),
        )
    ]
    if bias_old is not None or bias_new is not None:
      named.append(("bias_updates", bias_old, bias_new))

    values_ok = True
    for name, a, b in named:
      a, b = np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)
      diff = float(np.max(np.abs(b - a)))
      scale = float(np.max(np.abs(a)))
      rel_l2 = float(np.linalg.norm(b - a)) / max(float(np.linalg.norm(a)), 1e-30)
      if compare == "bitwise" or (compare == "reduction_order" and name in ("output", "loss", "bias_updates")):
        ok = bool(np.array_equal(a, b)) if compare == "bitwise" else diff <= _REL_TOL * max(scale, 1.0)
      elif compare == "reduction_order":
        ok = diff <= _REL_TOL * max(scale, 1e-6)
      else:
        ok = rel_l2 < _BF16_REL_NORM_TOL
      ok = ok and bool(np.isfinite(b).all())
      values_ok &= ok
    return structure_ok and values_ok, s_new

  @parameterized.named_parameters(
      ("real_f32_2chunks", {}, "real", "bitwise"),
      ("single_chunk", {"num_moe_token_chunks": 1}, "real", "bitwise"),
      ("padding_excluded", {"moe_exclude_padding_from_routing": True}, "padmask", "bitwise"),
      (
          "forced_with_fallback_and_dropping",
          {"ragged_buffer_factor": 1.0, "moe_fallback_topk_capacity": 8, "moe_exclude_padding_from_routing": True},
          "forced_padmask",
          "reduction_order",
      ),
      (
          "count_index_math",
          {"moe_routing_index_math": "count", "moe_exclude_padding_from_routing": True},
          "padmask",
          "bitwise",
      ),
      (
          "load_balance_and_routed_bias",
          {
              "model_name": "deepseek3-671b",
              "load_balance_loss_weight": 0.01,
              "routed_bias": True,
              "routed_bias_update_rate": 1e-3,
          },
          "padmask",
          "reduction_order",
      ),
      ("bf16_router", {"float32_gate_logits": False}, "real", "bf16"),
      ("ring_combine_bwd_rowwise", {"moe_quantize_combine_bwd_method": "rowwise"}, "real", "bf16"),
  )
  def test_parity_and_single_packed_gather(self, overrides, mode, compare):
    ok, _ = self._assert_case(overrides, mode=mode, compare=compare)
    self.assertTrue(ok)

  def test_moe_routing_maps_remat_device_and_offload(self):
    """Under jax.checkpoint, moe_routing_maps=device/offload skips recomputing the forward packed all-gather."""
    ok_remat, s_remat = self._assert_case({"moe_routing_maps": "remat"}, use_remat=True)
    ok_device, s_device = self._assert_case({"moe_routing_maps": "device"}, use_remat=True)
    ok_offload, s_offload = self._assert_case({"moe_routing_maps": "offload"}, use_remat=True)
    self.assertTrue(ok_remat and ok_device and ok_offload)

    self.assertEqual(s_remat["grad_packed_gathers"], 2)
    self.assertEqual(s_device["grad_packed_gathers"], 1)
    self.assertEqual(s_offload["grad_packed_gathers"], 1)
    self.assertEqual(s_remat["packed_host_puts"], 0)
    self.assertEqual(s_device["packed_host_puts"], 0)
    self.assertEqual(s_offload["packed_host_puts"], 1)
    self.assertEqual(s_offload["packed_device_puts"], 1)

  def test_gathered_token_mask_tagged_under_moe_routing_maps_with_aux_loss(self):
    """Under Switch aux loss + token_mask, gathered_token_mask is tagged and not re-all-gathered in backward."""
    aux_kw = {
        "load_balance_loss_weight": 0.01,
        "moe_use_megatron_seq_aux_loss": False,
        "moe_exclude_padding_from_routing": True,
    }
    ok_remat, s_remat = self._assert_case(
        {**aux_kw, "moe_routing_maps": "remat"}, mode="padmask", compare="reduction_order", use_remat=True
    )
    ok_device, s_device = self._assert_case(
        {**aux_kw, "moe_routing_maps": "device"}, mode="padmask", compare="reduction_order", use_remat=True
    )
    ok_offload, s_offload = self._assert_case(
        {**aux_kw, "moe_routing_maps": "offload"}, mode="padmask", compare="reduction_order", use_remat=True
    )
    self.assertTrue(ok_remat and ok_device and ok_offload)

    self.assertEqual(s_remat["mask_gathers"], 1)
    self.assertEqual(s_remat["mask_tags"], [])
    self.assertEqual(s_remat["grad_mask_gathers"], 2)
    self.assertEqual(s_remat["compiled_bool_all_gathers"], 2)

    self.assertEqual(s_device["mask_gathers"], 1)
    self.assertEqual(s_device["mask_tags"], ["moe_routing_maps"])
    self.assertEqual(s_device["grad_mask_gathers"], 1)
    self.assertEqual(s_device["compiled_bool_all_gathers"], 1)
    self.assertEqual(s_device["mask_host_puts"], 0)

    self.assertEqual(s_offload["mask_gathers"], 1)
    self.assertEqual(s_offload["mask_tags"], ["moe_routing_maps"])
    self.assertEqual(s_offload["grad_mask_gathers"], 1)
    self.assertEqual(s_offload["mask_host_puts"], 1)
    self.assertEqual(s_offload["mask_device_puts"], 1)

    # Negative control: skipping tag_fn on bool gathered_token_mask causes a duplicate backward bool all-gather.
    real_tag_fn = moe.routing_tag_fn

    def skip_bool_tag(config):
      fn = real_tag_fn(config)
      if fn is None:
        return None
      return lambda t: t if t.dtype == jnp.bool_ else fn(t)

    cfg_untagged = _cfg(**aux_kw, moe_routing_maps="device", moe_roe_packed_routing_gather=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_untagged), cfg_untagged.mesh_axes)
    x, forced, token_mask = _inputs(cfg_untagged, "padmask")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_untagged.logical_axis_rules):
      _, params, _ = nnx.split(_build(cfg_untagged, mesh), nnx.Param, ...)
    with mock.patch.object(moe, "routing_tag_fn", skip_bool_tag):
      _, s_untagged = self._run(cfg_untagged, mesh, params, x, forced, token_mask, use_remat=True)
    self.assertEqual(s_untagged["mask_tags"], [])
    self.assertEqual(s_untagged["grad_mask_gathers"], 2)
    self.assertEqual(s_untagged["compiled_bool_all_gathers"], 2)

  def test_negative_controls_are_caught(self):
    real_ag = jax.lax.all_gather

    def shifted_ag(x, axis_name, **kwargs):
      out = real_ag(x, axis_name, **kwargs)
      if out.ndim == 3 and out.shape[-1] == 2 * _TOP_K:
        out = jnp.roll(out, out.shape[1] // 2, axis=1)
      return out

    ok_shifted, _ = self._assert_case({}, control=mock.patch.object(jax.lax, "all_gather", shifted_ag))
    self.assertFalse(ok_shifted, "Shifted packed routing gather must fail parity check.")

    def zero_masked_mean(values, token_mask=None):
      del token_mask
      return jnp.zeros((values.shape[0],) + values.shape[2:], dtype=jnp.float32)

    ok_dropped_lb, _ = self._assert_case(
        {"load_balance_loss_weight": 0.01},
        compare="reduction_order",
        control=mock.patch.object(moe, "masked_sequence_mean", zero_masked_mean),
    )
    self.assertFalse(ok_dropped_lb, "Dropped load-balance probability statistic must fail parity check.")

  def test_default_flags_have_zero_jaxpr_and_hlo_drift(self):
    """When moe_topk_before_ep_all_gather=False and the routing flags are at defaults, HLO is untouched."""
    cfg = _cfg(moe_topk_before_ep_all_gather=False, moe_roe_packed_routing_gather=False)
    self.assertIs(cfg.moe_exclude_padding_from_routing, False)
    self.assertEqual(cfg.moe_fallback_topk_capacity, 0)
    self.assertIs(cfg.moe_roe_packed_routing_gather, False)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x, _, _ = _inputs(cfg, "real")
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      _, params, _ = nnx.split(_build(cfg, mesh), nnx.Param, ...)
    _, structure = self._run(cfg, mesh, params, x, None, None)
    self.assertEqual(structure["packed_gathers"], 0)
    self.assertEqual(structure["topk_gathers"], 0)


if __name__ == "__main__":
  unittest.main()

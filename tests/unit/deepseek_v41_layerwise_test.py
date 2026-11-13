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

"""CPU tests of the DeepSeek-V4.1 layerwise harness on the 14-layer mini twin (4 host devices).

Run with XLA_FLAGS=--xla_force_host_platform_device_count=4. The mini twin's weights go through the same path as the
real checkpoint: they are written as an HF safetensors checkpoint and streamed back by `lw_model.load_weights`.
"""

import copy
import dataclasses
import hashlib
import json
import pickle
import os
import struct
import time

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HAS_TORCH = sys.modules.get("torch", -1) is not None and importlib.util.find_spec("torch") is not None

if HAS_TORCH:
  import torch
  from safetensors.torch import save_file
else:
  torch = None
  save_file = None

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import yaml
from flax import nnx
from safetensors import safe_open

from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import maxtext_utils
from tests.utils.deepseek_v41_layerwise import lw_runs as lw_runs_mod
from tests.utils.deepseek_v41_layerwise import lw_stages

from tests.utils.deepseek_v41_layerwise import (  # isort: skip
    lw_bundle,
    lw_chain,
    lw_checks,
    lw_config,
    lw_l1,
    lw_l2,
    lw_l3,
    lw_l4,
    lw_model,
    lw_programs,
    lw_routing,
)

from tests.utils.parity import negctl

if HAS_TORCH:
  from tests.utils import deepseek_v41_mm_testing as mm_testing
  from tests.utils.deepseek_v41_reference import multimodal
  from tests.utils.parity import layer_parity
else:
  mm_testing = None
  multimodal = None
  layer_parity = lw_checks.layer_parity

_NUM_DEVICES = 4
_BATCH = 2
_TOL = lw_checks.load_tolerances(lw_checks.TOLERANCES_PATH)


def _mini_config(**overrides):
  """Builds a mini multimodal config for testing."""
  if not HAS_TORCH:
    raise unittest.SkipTest("PyTorch required for reference model")
  args = {
      "scan_layers": False,
      "remat_policy": "none",
      "ici_fsdp_parallelism": _NUM_DEVICES,
      "per_device_batch_size": _BATCH / _NUM_DEVICES,
  }
  args.update(overrides)
  return mm_testing.mini_vision_maxtext_config(**args)


class _MiniFixture:
  """The mini reference, its HF checkpoint and deterministic decoder inputs, built once per process."""

  _instance = None

  @classmethod
  def get(cls) -> "_MiniFixture":
    if not HAS_TORCH:
      raise unittest.SkipTest("PyTorch required for reference model")
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def __init__(self):
    self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with  # Lives for the whole process.
    self.ckpt = pathlib.Path(self.tmp.name) / "hf"
    args = multimodal.mini_model_args()
    self.ref = multimodal.build_multimodal_model(args, tokenizer=mm_testing.get_reference_tokenizer())
    multimodal.init_mini_reference(self.ref, seed=1)
    mm_testing.write_mini_safetensors_checkpoint(str(self.ckpt), self.ref, param_dtype=torch.float32)
    self.batch = multimodal.build_mini_vl_batch(args, seed=101, dtype=torch.float32)
    self.config = _mini_config()
    self.mesh = lw_config.build_mesh(self.config, jax.devices())
    rng = np.random.default_rng(7)
    seq = self.config.max_target_length
    self.tokens = self.batch.input_ids.numpy().astype(np.int32)
    self.image_mask = self.batch.image_mask.numpy().astype(bool)
    self.positions = np.broadcast_to(np.arange(seq, dtype=np.int32), (_BATCH, seq)).copy()
    self.h = rng.standard_normal((_BATCH, seq, self.config.mhc_expansion_rate, self.config.emb_dim)).astype(np.float32)
    pre = rng.standard_normal((_BATCH, seq, self.config.mhc_expansion_rate))
    self.pre_mix = (np.exp(pre) / np.exp(pre).sum(-1, keepdims=True)).astype(np.float32)

  def load_layer(self, config, mesh, layer_id: int) -> lw_model.ModuleParts:
    build = lw_programs.layer_build_fn(config, mesh, layer_id)
    parts = lw_model.abstract_parts(build, config, mesh)
    loaded = lw_model.load_weights(
        {"decoder": {f"layers_{layer_id}": nnx.to_pure_dict(parts.weights)}}, config, self.ckpt
    )
    parts = lw_model.fill_weights(parts, loaded["decoder"][f"layers_{layer_id}"])
    return lw_model.ModuleParts(parts.graphdef, parts.weights, lw_model.materialize_rest(build, config, mesh))

  def carry_for(self, config, layer_id: int, prefix_parts) -> dict:
    """The carry entering `layer_id`: the fixture's h / pre_mix run through the given earlier layers (natural)."""
    carry = {"h": jnp.asarray(self.h), "pre_mix": jnp.asarray(self.pre_mix)}
    for parts in prefix_parts:
      layer = parts.merge()
      carry = layer(carry, jnp.asarray(self.positions), jnp.asarray(self.tokens), image_mask=jnp.asarray(self.image_mask))
    reads = lw_programs.layer_reads(config, layer_id)
    return {k: v for k, v in carry.items() if k in ("h", "pre_mix") + reads}


def _natural_replay(program, parts, carry, fx):
  """Replay dict equal to the layer's own natural choice (two passes: the indexer choice feeds the MoE input)."""
  layer = parts.merge()
  spec = layer.spec
  seq = fx.config.max_target_length
  replay = {"moe_topk": jnp.zeros((_BATCH, seq, fx.config.num_experts_per_tok), jnp.int32)}
  if spec.is_index_source:
    replay["indexer_mask"] = jnp.ones((_BATCH, seq, seq // spec.compress_ratio), bool)
  fwd = jax.jit(lw_programs.group_forward_entry(program, capture=False))
  args = (
      [parts.weights],
      [parts.rest],
      carry,
      jnp.asarray(fx.positions),
      jnp.asarray(fx.tokens),
      jnp.asarray(fx.image_mask),
  )
  for _ in range(2):
    _, naturals, _ = fwd(*args, [replay])
    replay = dict(naturals[0])
  return replay


class ReplayedRoutingTest(unittest.TestCase):
  """Lead conditions (1)-(3) on the test-scoped copy of `RoutedMoE.deepseek_routing`."""

  def test_copied_routing_source_is_pinned(self):
    self.assertEqual(lw_routing.deepseek_routing_source_sha256(), lw_routing.ORIGINAL_DEEPSEEK_ROUTING_SHA256)

  def test_changed_routing_source_is_rejected(self):
    def changed(self, gate_logits, pre_bias_logits):  # pylint: disable=unused-argument
      return gate_logits, pre_bias_logits

    with mock.patch.object(moe.RoutedMoE, "deepseek_routing", changed):
      with self.assertRaisesRegex(AssertionError, "deepseek_routing changed"):
        with lw_routing.replayed_routing():
          pass
    with lw_routing.replayed_routing():
      self.assertIs(moe.RoutedMoE.deepseek_routing, lw_routing.replayed_deepseek_routing)

  def test_replaying_natural_choice_is_bitwise_unpatched_forward_and_grads(self):
    """Op-for-op (eager) execution, so the comparison sees the math of the copied routing, not XLA fusion choices."""
    fx = _MiniFixture.get()
    prefix = [fx.load_layer(fx.config, fx.mesh, i) for i in (0, 1)]
    parts = fx.load_layer(fx.config, fx.mesh, 2)
    carry = fx.carry_for(fx.config, 2, prefix)
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (2,), (parts.graphdef,), (None,))
    replay = _natural_replay(program, parts, carry, fx)
    positions, tokens, image_mask = jnp.asarray(fx.positions), jnp.asarray(fx.tokens), jnp.asarray(fx.image_mask)
    cot = jnp.asarray(np.random.default_rng(3).standard_normal(carry["h"].shape), jnp.float32)
    zero_indexer_cot = jnp.zeros(replay["indexer_mask"].shape, jnp.float32)

    def plain_loss(w, h, pre_mix):
      layer = lw_model.merge_fresh(parts.graphdef, w, parts.rest)
      out = layer({"h": h, "pre_mix": pre_mix}, positions, tokens, image_mask=image_mask)
      return jnp.sum(out["h"].astype(jnp.float32) * cot), out

    with jax.disable_jit():
      outs, naturals, _ = lw_programs.group_forward_entry(program, capture=False)(
          [parts.weights], [parts.rest], carry, positions, tokens, image_mask, [replay]
      )
      (_, plain_out), plain_grads = jax.value_and_grad(plain_loss, argnums=(0, 1, 2), has_aux=True)(
          parts.weights, carry["h"], carry["pre_mix"]
      )
      replay_grads = lw_programs.layer_vjp_entry(program)(
          parts.weights,
          parts.rest,
          carry,
          positions,
          tokens,
          image_mask,
          replay,
          {"h": cot, "pre_mix": jnp.zeros(carry["pre_mix"].shape, jnp.float32), "indexer_scores": zero_indexer_cot},
      )
    np.testing.assert_array_equal(np.asarray(outs[0]["h"]), np.asarray(plain_out["h"]))
    np.testing.assert_array_equal(np.asarray(outs[0]["pre_mix"]), np.asarray(plain_out["pre_mix"]))
    np.testing.assert_array_equal(np.asarray(naturals[0]["moe_topk"]), np.asarray(replay["moe_topk"]))
    flat_plain = lw_l3.unscanned_grad_leaves([plain_grads[0]])
    flat_replay = lw_l3.unscanned_grad_leaves([replay_grads[0]])
    self.assertEqual(set(flat_plain), set(flat_replay))
    self.assertIn("0/mlp/MoeBlock_0/wi_0", flat_plain)
    self.assertGreater(len(flat_plain), 20)
    for key in flat_plain:
      np.testing.assert_array_equal(np.asarray(flat_replay[key]), np.asarray(flat_plain[key]), err_msg=key)
    np.testing.assert_array_equal(np.asarray(replay_grads[1]), np.asarray(plain_grads[1]))
    np.testing.assert_array_equal(np.asarray(replay_grads[2]), np.asarray(plain_grads[2]))

  def test_indexer_cotangent_reaches_indexer_weights(self):
    fx = _MiniFixture.get()
    prefix = [fx.load_layer(fx.config, fx.mesh, i) for i in (0, 1)]
    parts = fx.load_layer(fx.config, fx.mesh, 2)
    carry = fx.carry_for(fx.config, 2, prefix)
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (2,), (parts.graphdef,), (None,))
    replay = _natural_replay(program, parts, carry, fx)
    io = (jnp.asarray(fx.positions), jnp.asarray(fx.tokens), jnp.asarray(fx.image_mask))
    vjp = jax.jit(lw_programs.layer_vjp_entry(program))
    zeros_h = jnp.zeros(carry["h"].shape, jnp.float32)
    zeros_pre = jnp.zeros(carry["pre_mix"].shape, jnp.float32)
    shape = replay["indexer_mask"].shape
    with_indexer = vjp(
        parts.weights,
        parts.rest,
        carry,
        *io,
        replay,
        {"h": zeros_h, "pre_mix": zeros_pre, "indexer_scores": jnp.ones(shape, jnp.float32)},
    )
    without = vjp(
        parts.weights,
        parts.rest,
        carry,
        *io,
        replay,
        {"h": zeros_h, "pre_mix": zeros_pre, "indexer_scores": jnp.zeros(shape, jnp.float32)},
    )
    grad_on = lw_l3.unscanned_grad_leaves([with_indexer[0]])["0/self_attention/indexer/weights_proj/kernel"]
    grad_off = lw_l3.unscanned_grad_leaves([without[0]])["0/self_attention/indexer/weights_proj/kernel"]
    self.assertGreater(float(jnp.max(jnp.abs(grad_on))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(grad_off))), 0.0)

  def test_corrupted_replay_and_bias_vl_flip_break_in_forward_agreement(self):
    fx = _MiniFixture.get()
    prefix = [fx.load_layer(fx.config, fx.mesh, i) for i in (0, 1)]
    parts = fx.load_layer(fx.config, fx.mesh, 2)
    carry = fx.carry_for(fx.config, 2, prefix)
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (2,), (parts.graphdef,), (None,))
    replay = _natural_replay(program, parts, carry, fx)
    fwd = jax.jit(lw_programs.group_forward_entry(program, capture=False))
    io = (jnp.asarray(fx.positions), jnp.asarray(fx.tokens), jnp.asarray(fx.image_mask))
    everywhere = np.ones((_BATCH, fx.config.max_target_length), bool)

    outs, naturals, _ = fwd([parts.weights], [parts.rest], carry, *io, [replay])
    clean = lw_routing.set_agreement(np.asarray(naturals[0]["moe_topk"]), np.asarray(replay["moe_topk"]), everywhere)
    self.assertEqual(clean.failures, 0)
    self.assertEqual(clean.eligible, everywhere.size)

    corrupted = dict(replay)
    corrupted["moe_topk"] = jnp.asarray(
        lw_routing.corrupt_replayed_expert(np.asarray(replay["moe_topk"]), (1, 7), fx.config.num_experts)
    )
    outs_a, naturals_a, _ = fwd([parts.weights], [parts.rest], carry, *io, [corrupted])
    nc_a = lw_routing.set_agreement(np.asarray(naturals_a[0]["moe_topk"]), np.asarray(corrupted["moe_topk"]), everywhere)
    self.assertEqual(nc_a.failure_positions, ((1, 7),))
    self.assertFalse(np.array_equal(np.asarray(outs_a[0]["h"])[1, 7], np.asarray(outs[0]["h"])[1, 7]))

    with lw_routing.gate_image_mask_flipped():
      outs_c, naturals_c, _ = jax.jit(lw_programs.group_forward_entry(program, capture=False))(
          [parts.weights], [parts.rest], carry, *io, [replay]
      )
    image_positions = fx.image_mask
    nc_c = lw_routing.set_agreement(
        np.asarray(naturals_c[0]["moe_topk"]), np.asarray(replay["moe_topk"]), image_positions
    )
    self.assertGreater(nc_c.failures, 0)
    np.testing.assert_array_equal(np.asarray(outs_c[0]["h"]), np.asarray(outs[0]["h"]))

  def test_margin_eligibility_applies_the_ulp_once_at_the_tie_multiplier(self):
    seq = 407
    kth = np.full((1, seq), 3.0, np.float32)
    boundary_ulp = np.abs(np.spacing(kth)).astype(np.float32)  # capture: `abs(np.spacing(s_k))`
    margins = np.where(np.arange(seq)[None] % 2 == 0, 9.0, 7.0).astype(np.float32) * boundary_ulp
    margins[0, 4] = np.nan  # no (k+1)-th competitor (an otherwise 9-ULP position)
    eligible = lw_routing.eligible_by_margin(margins, boundary_ulp, 8.0)
    self.assertEqual(eligible.shape, (1, seq))
    self.assertTrue(eligible[0, 0])  # 9 ULP
    self.assertFalse(eligible[0, 1])  # 7 ULP: a tie under the 8-ULP rule
    self.assertFalse(eligible[0, 4])
    self.assertEqual(int(eligible.sum()), (seq + 1) // 2 - 1)
    with self.assertRaisesRegex(ValueError, "differ in shape"):
      lw_routing.eligible_by_margin(margins, boundary_ulp[:, :10], 8.0)

  def test_indexer_geometry_status(self):
    positions = np.arange(1024)[None, :]
    ratio2 = lw_routing.indexer_geometry(positions, 2, 512)
    ratio1 = lw_routing.indexer_geometry(positions, 1, 512)
    self.assertEqual(ratio2["geometrically_eligible_queries"], 0)
    self.assertEqual(ratio1["geometrically_eligible_queries"], 1024 - 512)
    empty = lw_routing.Agreement(eligible=0, failures=0, failure_positions=())
    self.assertEqual(
        lw_routing.agreement_status(empty, geometrically_eligible=0, allowed_failures=0), "NOT_APPLICABLE_GEOMETRY"
    )
    self.assertEqual(lw_routing.agreement_status(empty, geometrically_eligible=512, allowed_failures=0), "FAIL")
    good = lw_routing.Agreement(eligible=10, failures=0, failure_positions=())
    self.assertEqual(lw_routing.agreement_status(good, geometrically_eligible=512, allowed_failures=0), "PASS")


class ChecksTest(unittest.TestCase):

  def test_tolerances_file_is_the_jax_ml_parity_ceilings(self):
    self.assertEqual(_TOL.code_parity, layer_parity.FP32_TOLERANCE)
    self.assertEqual(_TOL.fp64_frob_rel, layer_parity.FP64_FROB_REL_CEILING)
    self.assertEqual(_TOL.decoder_layer_fp64_status, lw_checks.DECODER_LAYER_FP64_STATUS)
    self.assertGreaterEqual(_TOL.negative_control_rel_scale, negctl.REL_NORM_BAND[0])
    self.assertLessEqual(_TOL.negative_control_rel_scale, negctl.REL_NORM_BAND[1])

  def test_tolerances_reject_missing_extra_and_loosened_keys(self):
    good = json.loads(lw_checks.TOLERANCES_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
      path = pathlib.Path(tmp) / "t.json"
      l3 = good["l3_invariance"]
      gap = good["tier_b_reference_gap_conditioning"]
      cases = {
          "keys": {k: v for k, v in good.items() if k != "l3_invariance"},
          "keys ": {**good, "dead_block": {}},
          "exceeds approved fp32 ceilings": {**good, "code_parity": {**good["code_parity"], "frob_rel": 2.5e-5}},
          "code_parity keys": {**good, "code_parity": {**good["code_parity"], "extra": 1.0}},
          "l3_invariance keys": {**good, "l3_invariance": {k: v for k, v in l3.items() if k != "scan_vs_unscan_max_abs"}},
          "l3_invariance keys ": {**good, "l3_invariance": {**l3, "dead": 0.0}},
          "scan_vs_unscan_max_abs = 1.0 is outside": {**good, "l3_invariance": {**l3, "scan_vs_unscan_max_abs": 1.0}},
          "remat_forward_max_abs = 1e-07 is outside": {**good, "l3_invariance": {**l3, "remat_forward_max_abs": 1e-7}},
          "remat_backward_max_abs = 1e-06 is outside": {**good, "l3_invariance": {**l3, "remat_backward_max_abs": 1e-6}},
          "conditioning keys": {**good, "tier_b_reference_gap_conditioning": {**gap, "dead": 0}},
          "min_cos_sim = 0.9 is outside": {**good, "tier_b_reference_gap_conditioning": {**gap, "min_cos_sim": 0.9}},
          "max_rel_l2 = 0.5 is outside": {**good, "tier_b_reference_gap_conditioning": {**gap, "max_rel_l2": 0.5}},
          "decoder_layer_fp64_status = 'none' != 'f32-cast-limited'": {
              **good,
              "decoder_layer_fp64_status": "none",
          },
          "negative_control_rel_scale = 0.0 is outside": {**good, "negative_control_rel_scale": 0.0},
          "negative_control_rel_scale = 0.5 is outside": {**good, "negative_control_rel_scale": 0.5},
      }
      for pattern, data in cases.items():
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex((ValueError, KeyError), pattern.strip()):
          lw_checks.load_tolerances(path)
      path.write_text(json.dumps(good), encoding="utf-8")
      self.assertEqual(lw_checks.load_tolerances(path).code_parity, layer_parity.FP32_TOLERANCE)

  def test_production_check_needs_every_bound_key_and_a_defined_cosine(self):
    ref = np.linspace(1.0, 2.0, 64)
    bound = {"max_abs_diff": 0.1, "rel_l2": 0.01, "min_cos_sim": 0.999}
    self.assertTrue(lw_checks.check_production("x", ref + 1e-4, ref, bound).passed)
    self.assertFalse(lw_checks.check_production("x", ref + 0.5, ref, bound).passed)
    with self.assertRaisesRegex(KeyError, "min_cos_sim"):
      lw_checks.check_production("x", ref, ref, {"max_abs_diff": 0.1, "rel_l2": 0.01})
    with self.assertRaisesRegex(ValueError, "norm below"):
      lw_checks.check_production("x", np.zeros(8), np.zeros(8), bound)

  def test_sha256sums_verification(self):
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      (d / "a.npz").write_bytes(b"alpha")
      digest = lw_checks.sha256_file(d / "a.npz")
      (d / "SHA256SUMS.txt").write_text(f"{digest}  a.npz\n", encoding="utf-8")
      self.assertEqual(lw_checks.verify_sha256sums(d, ["a.npz"]), {"a.npz": digest})
      with self.assertRaisesRegex(KeyError, "b.npz"):
        lw_checks.verify_sha256sums(d, ["b.npz"])
      (d / "a.npz").write_bytes(b"tampered")
      with self.assertRaisesRegex(ValueError, "pins"):
        lw_checks.verify_sha256sums(d, ["a.npz"])

  def test_l3_moe_implementations_run_without_activation_quant_and_only_that_override_is_accepted(self):
    variants = lw_l3.moe_variants(_NUM_DEVICES)
    for name in ("dense_natural", "sparse_megablox", "sparse_ragged"):
      self.assertIs(variants[name].config_overrides()[lw_config.L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE], False)
    for name in ("dense_ep1", "dense_epn"):
      self.assertNotIn(lw_config.L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE, variants[name].config_overrides())
    self.assertEqual(len({v.program_name() for v in variants.values()}), len(variants))
    common = {
        "tokenizer_path": ".",
        "seq_len": 1024,
        "batch_size": 1,
        "num_devices": _NUM_DEVICES,
        "expert_parallelism": 1,
    }
    with self.assertRaisesRegex(ValueError, "may only be overridden to False"):
      lw_config.build_config(
          lw_config.TIER_CODE_PARITY, reference_activation_quant=False, extra={"ds41_activation_quant": True}, **common
      )
    with self.assertRaisesRegex(ValueError, "would override"):
      lw_config.build_config(
          lw_config.TIER_CODE_PARITY, reference_activation_quant=True, extra={"dtype": "bfloat16"}, **common
      )

  def test_both_tiers_disable_activation_quant_matching_reference_identity_kernel(self):
    prod_flags = lw_config.tier_flags(lw_config.TIER_PRODUCTION, reference_activation_quant=True)
    self.assertEqual(prod_flags, dict(lw_config.W1_R4_PRODUCTION_FLAGS))
    self.assertFalse(prod_flags["ds41_activation_quant"])
    code_parity_true = lw_config.tier_flags(lw_config.TIER_CODE_PARITY, reference_activation_quant=True)
    code_parity_false = lw_config.tier_flags(lw_config.TIER_CODE_PARITY, reference_activation_quant=False)
    self.assertFalse(code_parity_true["ds41_activation_quant"])
    self.assertFalse(code_parity_false["ds41_activation_quant"])

  def test_tier_b_check_fails_on_an_ill_conditioned_reference_gap(self):
    rng = np.random.default_rng(2)
    fp32 = rng.standard_normal(256)
    good_bound = {
        "max_abs_diff": 1.0,
        "rel_l2": 0.1,
        "min_cos_sim": 0.99,
        "measured_min_cos_sim": 0.9999,
        "measured_worst_rel_l2": 0.01,
    }
    bounds = lw_l2.GateBounds(lw_config.TIER_PRODUCTION, _TOL, {"all_positions": good_bound}, 0, 0, 2.5)
    ref = lw_l2.ReferenceTensor(fp64=None, fp32=fp32, bf16=fp32 + 1e-3 * rng.standard_normal(256))
    ok = lw_l2.compare_tensor("x", fp32 + 1e-3, ref, None, bounds, "all_positions")
    self.assertEqual(ok["status"], "PASS")
    self.assertNotIn("condition", ok)
    bad_bounds = dataclasses.replace(bounds, production={"all_positions": {**good_bound, "measured_min_cos_sim": -0.39}})
    bad = lw_l2.compare_tensor("x", fp32 + 1e-3, ref, None, bad_bounds, "all_positions")
    self.assertEqual((bad["status"], bad["ill_conditioned_gaps"]), ("ILL_CONDITIONED_REF_GAP", ["bound"]))
    noisy_ref = dataclasses.replace(ref, bf16=fp32 + 0.5 * rng.standard_normal(256))
    noisy = lw_l2.compare_tensor("x", fp32 + 1e-3, noisy_ref, None, bounds, "all_positions")
    self.assertEqual((noisy["status"], noisy["ill_conditioned_gaps"]), ("ILL_CONDITIONED_REF_GAP", ["tensor"]))
    with self.assertRaisesRegex(KeyError, "pre_mix_out"):
      lw_l2.compare_tensor("x", fp32 + 1e-3, ref, None, bounds, "pre_mix_out")

  def test_l3_max_abs_check_reports_rel_l2_and_gates_on_max_abs(self):
    ref = np.linspace(1.0, 2.0, 64, dtype=np.float32)
    result = lw_checks.check_max_abs("x", ref + np.float32(1e-3), ref, 1e-2)
    self.assertTrue(result.passed)
    self.assertAlmostEqual(result.metrics["rel_l2"], 1e-3 * 8.0 / np.linalg.norm(ref.astype(np.float64)), places=6)
    self.assertFalse(lw_checks.check_max_abs("x", ref + np.float32(1e-1), ref, 1e-2).passed)
    zero = lw_checks.check_max_abs("z", np.zeros(4, np.float32), np.zeros(4, np.float32), 0.0)
    self.assertTrue(zero.passed)
    self.assertIsNone(zero.metrics["rel_l2"])
    self.assertEqual(zero.metrics["ref_l2_norm"], 0.0)

  def test_control_fired_and_agreement_fired_require_clean_pass_turning_to_fail(self):
    clean_fail = [{"label": "a", "status": "FAIL"}]
    perturbed_fail = [{"label": "a", "status": "FAIL"}]
    self.assertFalse(lw_checks.control_fired(clean_fail, perturbed_fail))
    clean_pass = [{"label": "a", "status": "PASS"}]
    self.assertFalse(lw_checks.control_fired(clean_pass, clean_pass))
    self.assertTrue(lw_checks.control_fired(clean_pass, perturbed_fail))
    self.assertTrue(
        lw_checks.control_fired(
            [{"label": "a", "status": "FAIL"}, {"label": "b", "status": "PASS"}],
            [{"label": "a", "status": "FAIL"}, {"label": "b", "status": "FAIL"}],
        )
    )
    self.assertFalse(lw_checks.control_fired(clean_pass, [{"label": "missing", "status": "FAIL"}]))
    self.assertFalse(lw_l2._agreement_fired("FAIL", "FAIL"))  # pylint: disable=protected-access
    self.assertFalse(lw_l2._agreement_fired("NOT_APPLICABLE_GEOMETRY", "FAIL"))  # pylint: disable=protected-access
    self.assertFalse(lw_l2._agreement_fired("PASS", "PASS"))  # pylint: disable=protected-access
    self.assertTrue(lw_l2._agreement_fired("PASS", "FAIL"))  # pylint: disable=protected-access

  def test_r5_defect1_measured_noise_floors_never_clamp_min_cos_upward_or_rel_l2_downward(self):
    live_floors = pathlib.Path("/home/yaoyuchen_google_com/ds41_mm_scratch/layerwise_ref/measured_noise_floors.json")
    if not live_floors.is_file():
      return
    nf = json.loads(live_floors.read_text(encoding="utf-8"))
    rec = nf["recommended_tolerances"]
    checked = 0
    for stage_key in ("L1_vision_aligner", "L2_decoder_groups", "L4_backward"):
      for tier in ("code_parity", "production"):
        for k, b in rec[stage_key][tier].items():
          if not isinstance(b, dict):
            continue
          self.assertLessEqual(
              b["min_cos_sim"], b["measured_min_cos_sim"] + 1e-12, f"{stage_key}.{tier}.{k} upward min_cos_sim clamp"
          )
          self.assertGreaterEqual(
              b["rel_l2"] + 1e-12, b["measured_worst_rel_l2"], f"{stage_key}.{tier}.{k} downward rel_l2 clamp"
          )
          self.assertGreaterEqual(
              b["max_abs_diff"] + 1e-12,
              b["measured_worst_max_abs_diff"],
              f"{stage_key}.{tier}.{k} downward max_abs_diff clamp",
          )
          checked += 1
    self.assertEqual(checked, 54)
    self.assertIn("dW_aligner", rec["L4_backward"]["production"])
    self.assertIn("dW_frozen_vit", rec["L4_backward"]["production"])
    self.assertLess(rec["L4_backward"]["production"]["dW_dense"]["measured_worst_rel_l2"], 0.25)
    self.assertLess(rec["L4_backward"]["production"]["dx"]["measured_worst_rel_l2"], 0.10)

  def test_r5_defect2_multi_block_conditioning_and_frozen_vit_chaotic_vs_ref_bf16(self):
    rng = np.random.default_rng(7)
    fp32 = rng.standard_normal(256)
    bf16 = fp32 + 0.12 * rng.standard_normal(256)
    gap = lw_checks.bf16_gap_metrics(bf16, fp32)
    mod_bound = {
        "max_abs_diff": 2.0,
        "rel_l2": 2.5 * gap["rel_l2"],
        "min_cos_sim": 1.0 - 2.5 * (1.0 - gap["cos_sim"]),
        "measured_min_cos_sim": gap["cos_sim"],
        "measured_worst_rel_l2": gap["rel_l2"],
    }
    bounds = lw_l2.GateBounds(lw_config.TIER_PRODUCTION, _TOL, {"vit_block": mod_bound}, 0, 0, 2.5)
    ref = lw_l2.ReferenceTensor(fp64=None, fp32=fp32, bf16=bf16)
    res = lw_l2.compare_tensor("vit_block_21", bf16, ref, None, bounds, "vit_block")
    self.assertEqual(res["status"], "PASS")
    self.assertFalse(res["passes_single_op_0_10_gap"])
    self.assertAlmostEqual(res["rel_l2_vs_ref_bf16"], 0.0, places=7)

    chaotic_bf16 = -0.3 * fp32 + 2.0 * rng.standard_normal(256)
    chaotic_gap = lw_checks.bf16_gap_metrics(chaotic_bf16, fp32)
    chaotic_bound = {
        "max_abs_diff": 10.0,
        "rel_l2": 2.5 * chaotic_gap["rel_l2"],
        "min_cos_sim": -1.0,
        "measured_min_cos_sim": chaotic_gap["cos_sim"],
        "measured_worst_rel_l2": chaotic_gap["rel_l2"],
    }
    chaotic_bounds = lw_l2.GateBounds(lw_config.TIER_PRODUCTION, _TOL, {"dW_frozen_vit": chaotic_bound}, 0, 0, 2.5)
    chaotic_ref = lw_l2.ReferenceTensor(fp64=None, fp32=fp32, bf16=chaotic_bf16)
    ok_chaotic = lw_l2.compare_tensor(
        "vision.blocks.0.attn.qkv.weight", chaotic_bf16, chaotic_ref, None, chaotic_bounds, "dW_frozen_vit"
    )
    self.assertEqual(ok_chaotic["status"], "PASS")
    self.assertTrue(ok_chaotic["frozen_vit_32block_bf16_chaotic"])
    bad_chaotic = lw_l2.compare_tensor(
        "vision.blocks.0.attn.qkv.weight",
        chaotic_bf16 + 0.5 * rng.standard_normal(256),
        chaotic_ref,
        None,
        chaotic_bounds,
        "dW_frozen_vit",
    )
    self.assertEqual(bad_chaotic["status"], "FAIL")

  def test_r5_defect3_near_zero_reference_norm_gates_by_max_abs(self):
    zeros = np.zeros((4, 8), dtype=np.float32)
    ref_a = lw_l2.ReferenceTensor(fp64=zeros.astype(np.float64), fp32=zeros, bf16=None)
    bounds_a = lw_l2.GateBounds(lw_config.TIER_CODE_PARITY, _TOL, None, 0, 0, None)
    ok_a = lw_l2.compare_tensor("d_pre_mix_in", zeros + np.float32(1e-6), ref_a, None, bounds_a, "d_pre_mix_in")
    self.assertEqual(ok_a["status"], "PASS")
    self.assertTrue(ok_a["near_zero_ref_norm_abs_gated"])
    fail_a = lw_l2.compare_tensor("d_pre_mix_in", zeros + np.float32(1e-3), ref_a, None, bounds_a, "d_pre_mix_in")
    self.assertEqual(fail_a["status"], "FAIL")

  def test_r5_defect4_tier_a_code_parity_dual_reports_1e5_and_measured_floor(self):
    rng = np.random.default_rng(11)
    fp64 = rng.standard_normal(128).astype(np.float64)
    fp32 = (fp64 + 5e-4 * rng.standard_normal(128)).astype(np.float32)
    ref = lw_l2.ReferenceTensor(fp64=fp64, fp32=fp32, bf16=None)
    cp_bound = {
        "max_abs_diff": 1e-2,
        "rel_l2": 5e-3,
        "min_cos_sim": 0.9999,
        "measured_min_cos_sim": 0.99999,
        "measured_worst_rel_l2": 1e-3,
    }
    bounds = lw_l2.GateBounds(lw_config.TIER_CODE_PARITY, _TOL, None, 0, 0, None, code_parity={"dW_dense": cp_bound})
    ok = lw_l2.compare_tensor("layers.0.self_attn.o_proj.weight", fp32, ref, None, bounds, "dW_dense")
    self.assertEqual(ok["status"], "PASS")
    self.assertTrue(ok["passes_vs_fp32_1e5"])
    self.assertFalse(ok["passes_1e5_vs_fp64"])
    self.assertTrue(ok["passes_adopted_vs_fp64"])
    bad = lw_l2.compare_tensor("layers.0.self_attn.o_proj.weight", fp32 + np.float32(1e-2), ref, None, bounds, "dW_dense")
    self.assertEqual(bad["status"], "FAIL")
    self.assertFalse(bad["passes_vs_fp32_1e5"])

  def test_r6_tier_a_passes_when_vs_fp32_exceeds_1e5_but_within_adopted_floor_and_fires_negative_control(self):
    rng = np.random.default_rng(19)
    fp64 = rng.standard_normal(256).astype(np.float64)
    delta_dir = rng.standard_normal(256).astype(np.float64)
    fp32 = (fp64 + 4e-4 * delta_dir).astype(np.float32)
    ref = lw_l2.ReferenceTensor(fp64=fp64, fp32=fp32, bf16=None)
    floor_metrics = lw_checks.check_code_parity("floor", fp32.astype(np.float64), fp64, _TOL.code_parity).metrics
    cp_bound = {
        "max_abs_diff": 5.0 * float(floor_metrics["max_abs"]),
        "rel_l2": 5.0 * float(floor_metrics["frob_rel"]),
        "min_cos_sim": 0.999,
        "measured_min_cos_sim": 0.99999,
        "measured_worst_rel_l2": float(floor_metrics["frob_rel"]),
    }
    bounds = lw_l2.GateBounds(lw_config.TIER_CODE_PARITY, _TOL, None, 0, 0, None, code_parity={"dW_frozen_vit": cp_bound})
    # Candidate is 5.2x the fp32-vs-fp64 floor away from fp64 (and > 1e-5 away from fp32):
    # res_vs_fp32.passed is False and 5.0x cp_bound is exceeded, but 5.5x floor_rel / 6.5x floor_abs passes.
    cand = (fp64 - 5.2 * (fp32.astype(np.float64) - fp64)).astype(np.float32)
    res = lw_l2.compare_tensor("vision.blocks.0.attn.wo.bias", cand, ref, None, bounds, "dW_frozen_vit")
    self.assertTrue(res["passed"])
    self.assertEqual(res["status"], "PASS")
    self.assertFalse(res["passes_vs_fp32_1e5"])
    self.assertFalse(res["passes_1e5_vs_fp64"])
    self.assertTrue(res["passes_adopted_vs_fp64"])
    self.assertEqual(res["original_ceiling_1e5"], _TOL.code_parity.rtol)
    self.assertAlmostEqual(res["adopted_ceiling_vs_fp64"], 5.5 * float(floor_metrics["frob_rel"]), places=10)
    for required_key in ("vs_fp32", "vs_fp64_floor_metrics", "reference_floor_fp32_vs_fp64"):
      self.assertIn(required_key, res)

    # Negative control: perturbing candidate by 1e-2 relative norm fails the check and fires control_fired.
    pert_cand = cand + np.float32(1e-2 * np.linalg.norm(cand) / np.sqrt(cand.size))
    pert_res = lw_l2.compare_tensor("vision.blocks.0.attn.wo.bias", pert_cand, ref, None, bounds, "dW_frozen_vit")
    self.assertFalse(pert_res["passed"])
    self.assertEqual(pert_res["status"], "FAIL")
    self.assertFalse(pert_res["passes_adopted_vs_fp64"])
    self.assertTrue(lw_checks.control_fired([res], [pert_res]))


class BundleTest(unittest.TestCase):

  def test_bundle_reads_are_checked_against_the_index(self):
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      x = np.arange(6, dtype=np.float32).reshape(2, 3)
      np.savez(d / "b.npz", x=x, x_fp64=x.astype(np.float64))
      keys = {"x": {"shape": [2, 3], "dtype": "float32"}, "x_fp64": {"shape": [2, 3], "dtype": "float64"}}
      entry = {"sha256": lw_checks.sha256_file(d / "b.npz"), "keys": keys}
      (d / "SHA256SUMS.txt").write_text(f"{entry['sha256']}  b.npz\n", encoding="utf-8")
      index = lw_bundle.BundleIndex(root=d, sha256="test-index", data={"bundles": {"b.npz": entry}})
      tier_a = index.open("b.npz").tensor("x", lw_config.TIER_CODE_PARITY)
      np.testing.assert_array_equal(tier_a.fp32, x)
      self.assertEqual(tier_a.fp64.dtype, np.float64)
      with self.assertRaisesRegex(KeyError, "x_bf16"):
        index.open("b.npz").tensor("x", lw_config.TIER_PRODUCTION)
      with self.assertRaisesRegex(KeyError, "'y'"):
        index.open("b.npz").get("y")
      with self.assertRaisesRegex(KeyError, "c.npz"):
        index.open("c.npz")
      wrong_shape = {**entry, "keys": {**keys, "x": {"shape": [3, 2], "dtype": "float32"}}}
      with self.assertRaisesRegex(ValueError, "index says"):
        lw_bundle.BundleFile(d / "b.npz", wrong_shape).get("x")
      with self.assertRaisesRegex(ValueError, "bundle_index.json pins"):
        lw_bundle.BundleFile(d / "b.npz", {**entry, "sha256": "0" * 64})

  def test_noise_floors_are_verified_against_sha256sums(self):
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      floors_json = {"recommended_tolerances": {"safety_multiplier_bf16": 2.5}}
      (d / "measured_noise_floors.json").write_text(json.dumps(floors_json), encoding="utf-8")
      digest = lw_checks.sha256_file(d / "measured_noise_floors.json")
      (d / "SHA256SUMS.txt").write_text(f"{digest}  measured_noise_floors.json\n", encoding="utf-8")
      floors = lw_bundle.load_noise_floors(d)
      self.assertEqual(floors.sha256, digest)
      self.assertEqual(floors.bf16_gap_multiplier(), 2.5)
      (d / "SHA256SUMS.txt").write_text(f"{'0' * 64}  measured_noise_floors.json\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "SHA256SUMS.txt pins"):
        lw_bundle.load_noise_floors(d)

  def test_reference_activation_quant_is_read_per_variant_and_required(self):
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      rel = lw_bundle.l1_path("M4")
      (d / rel).parent.mkdir(parents=True, exist_ok=True)
      np.savez(
          d / rel,
          reference_activation_quant_fp64=np.asarray(True),
          reference_activation_quant_fp32=np.asarray(False),
      )
      entry = {
          "sha256": lw_checks.sha256_file(d / rel),
          "keys": {
              "reference_activation_quant_fp64": {"shape": [], "dtype": "bool"},
              "reference_activation_quant_fp32": {"shape": [], "dtype": "bool"},
          },
      }
      (d / "SHA256SUMS.txt").write_text(f"{entry['sha256']}  {rel}\n", encoding="utf-8")
      index = lw_bundle.BundleIndex(root=d, sha256="test-index", data={"bundles": {rel: entry}})
      self.assertTrue(index.reference_activation_quant("M4", lw_bundle.TARGET_VARIANT[lw_config.TIER_CODE_PARITY]))
      self.assertFalse(index.reference_activation_quant("M4", lw_bundle.TARGET_VARIANT[lw_config.TIER_PRODUCTION]))
      with self.assertRaisesRegex(ValueError, "expected 'fp64' or 'fp32'"):
        index.reference_activation_quant("M4", "bf16")
      empty_index = lw_bundle.BundleIndex(
          root=d, sha256="test-index", data={"bundles": {rel: {"sha256": entry["sha256"], "keys": {}}}}
      )
      with self.assertRaisesRegex(KeyError, "reference_activation_quant_fp32"):
        empty_index.reference_activation_quant("M4", "fp32")

  def test_data_pins_outside_harness_dir_and_bundle_key_schema_matches_adapters(self):
    self.assertFalse(lw_bundle.DATA_PINS_PATH.is_relative_to(lw_checks.HARNESS_DIR))
    pins = lw_bundle.load_data_pins().data
    self.assertEqual(pins["schema_version"], 1)
    self.assertEqual(len(pins["bundle_index_sha256"]), 64)
    schema = pins["bundle_key_schema"]
    self.assertEqual(set(schema), {"l1_forward", "l1_vjp", "l2_layer", "l4_layer"})
    self.assertNotIn("candidates_mask_out", json.dumps(schema))
    self.assertNotIn("candidates_mask_out", (lw_checks.HARNESS_DIR / "lw_bundle.py").read_text(encoding="utf-8"))
    l2_tri = set(schema["l2_layer"]["tri_variant_keys"]) | set(
        schema["l2_layer"]["conditional_tri_variant_keys"]["has_engram"]
    )
    self.assertTrue(
        {f"{p}{pt[2]}" for p in ("{p}",) for pt in lw_l2.LOCALIZATION_POINTS} <= l2_tri,
    )

  def test_r5_addendum_bundle_index_resolves_canonical_fallback_and_verifies_sha256sums_lazily(self):
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      x = np.ones((2, 2), dtype=np.float32)
      canonical = "L4_vjp_vision_aligner_M4.npz"
      rel = lw_bundle.l1_vjp_path("M4")
      np.savez(d / canonical, d_image_input=x, d_image_input_fp64=x.astype(np.float64))
      digest = lw_checks.sha256_file(d / canonical)
      entry = {
          "sha256": digest,
          "keys": {
              "d_image_input": {"shape": [2, 2], "dtype": "float32"},
              "d_image_input_fp64": {"shape": [2, 2], "dtype": "float64"},
          },
      }
      other_rel = "L2_decoder_groups_M4.npz"
      np.savez(d / other_rel, d_image_input=x * 2.0, d_image_input_fp64=(x * 2.0).astype(np.float64))
      other_digest = lw_checks.sha256_file(d / other_rel)
      other_entry = {"sha256": other_digest, "keys": entry["keys"]}
      missing_rel = "ref_bundles/M4/l2_layer_00.npz"
      missing_sha = "a" * 64
      missing_entry = {"sha256": missing_sha, "keys": entry["keys"]}
      bundles = {
          canonical: entry,
          rel: entry,
          other_rel: other_entry,
          missing_rel: missing_entry,
      }
      (d / "SHA256SUMS.txt").write_text(
          f"{digest}  {canonical}\n"
          f"{digest}  {rel}\n"
          f"{other_digest}  {other_rel}\n"
          f"{missing_sha}  {missing_rel}\n"
          f"{'f' * 64}  unopened_intermediate.npz\n",
          encoding="utf-8",
      )
      index = lw_bundle.BundleIndex(root=d, sha256="test-idx", data={"bundles": bundles})
      opened = index.open(rel)
      np.testing.assert_array_equal(opened.get("d_image_input"), x)
      with self.assertRaises(FileNotFoundError):
        index.open(missing_rel)
      (d / "SHA256SUMS.txt").write_text(f"{digest}  {canonical}\n{'0' * 64}  {rel}\n", encoding="utf-8")
      index_bad_rel = lw_bundle.BundleIndex(root=d, sha256="test-idx", data={"bundles": bundles})
      with self.assertRaisesRegex(ValueError, "SHA256SUMS.txt pins"):
        index_bad_rel.open(rel)
      (d / "SHA256SUMS.txt").write_text(f"{'0' * 64}  {canonical}\n{digest}  {rel}\n", encoding="utf-8")
      index_bad_canon = lw_bundle.BundleIndex(root=d, sha256="test-idx", data={"bundles": bundles})
      with self.assertRaisesRegex(ValueError, "SHA256SUMS.txt pins"):
        index_bad_canon.open(rel)
      (d / "SHA256SUMS.txt").unlink()
      index_no_sums = lw_bundle.BundleIndex(root=d, sha256="test-idx", data={"bundles": bundles})
      with self.assertRaisesRegex(FileNotFoundError, "SHA256SUMS.txt"):
        index_no_sums.open(rel)


class StageWiringTest(unittest.TestCase):

  def test_aot_record_must_match_tier_geometry_and_layers_and_fit(self):
    record = {
        "program": "L4.layer[20]",
        "tier": "production",
        "sample": "M4",
        "seq_len": 1024,
        "layers": [20],
        "activation_quant": True,
        "harness_code_sha256": lw_checks.harness_code_sha256(),
        "model_code_sha256": lw_checks.model_code_sha256(),
        "compiled": True,
        "fits_device_hbm": True,
    }
    unfit = {**record, "tier": "code_parity", "fits_device_hbm": False}
    stale = {**record, "tier": "code_parity", "layers": [14], "harness_code_sha256": "0" * 64}
    stale_model = {**record, "tier": "code_parity", "layers": [24], "model_code_sha256": "1" * 64}
    aq_on = mock.Mock(ds41_activation_quant=True)
    run = lw_stages.StageRun(
        stage="L4",
        tier=lw_config.TIER_PRODUCTION,
        sample="M4",
        target="20",
        index=lw_bundle.BundleIndex(root=pathlib.Path("."), sha256="test-index", data={}),
        hf_dir=pathlib.Path("."),
        tokenizer_path=".",
        tolerances=_TOL,
        floors=lw_bundle.NoiseFloors(sha256="test-floors", data={}),
        aot=lw_checks.PinnedJson(
            path="aot.json", sha256="test-aot", data={"programs": [record, unfit, stale, stale_model]}
        ),
        expected_na=lw_stages.ExpectedNotApplicable(sha256="test-na", cells={}),
        memory=[],
    )
    name = lw_programs.layer_vjp_program_name(20)
    match = {"sample": "M4", "seq_len": 1024, "layers": [20]}
    self.assertIs(lw_stages.require_aot(run, name, aq_on, match), record)
    with self.assertRaisesRegex(ValueError, "has 0 records"):
      lw_stages.require_aot(run, name, aq_on, {**match, "layers": [14]})
    with self.assertRaisesRegex(ValueError, "has 0 records"):
      lw_stages.require_aot(run, name, mock.Mock(ds41_activation_quant=False), match)
    code_parity = dataclasses.replace(run, tier=lw_config.TIER_CODE_PARITY)
    with self.assertRaisesRegex(ValueError, "did not compile or fit"):
      lw_stages.require_aot(code_parity, name, aq_on, {"layers": [20]})
    with self.assertRaisesRegex(ValueError, "compiled from harness code 0000"):
      lw_stages.require_aot(code_parity, name, aq_on, {"layers": [14]})
    with self.assertRaisesRegex(ValueError, "compiled from model code 1111"):
      lw_stages.require_aot(code_parity, name, aq_on, {"layers": [24]})

  def test_run_verdict_counts_statuses_and_accepts_only_pinned_not_applicable_cells(self):
    geometry = {"compress_ratio": 2, "index_topk": 512, "max_reachable": 512, "geometrically_eligible_queries": 0}
    na = {"check": "indexer_agreement", "layer": 2, "geometry": geometry, "status": "NOT_APPLICABLE_GEOMETRY"}
    run = mock.Mock(stage="L2", sample="M4")
    run.expected_na = lw_stages.ExpectedNotApplicable(sha256="t", cells={("L2", "M4", 2, "indexer_agreement"): geometry})

    def report(*extra_checks, controls=None):
      checks = [{"label": "layer_2.h_out", "status": "PASS"}, *extra_checks]
      return {
          "status": "PASS",
          "layers": {"layer_2": {"checks": checks, "agreement": {"indexer": na}}},
          "negative_controls": controls or {"NC_L2b": {"status": "FIRED", "check": {"label": "x", "status": "FAIL"}}},
      }

    ok = lw_stages.finalize_report(report(), run)
    self.assertEqual(ok["status"], "PASS")
    self.assertEqual(ok["status_counts"]["PASS"], 1)
    self.assertEqual(ok["status_counts"]["NOT_APPLICABLE_GEOMETRY"], 1)
    self.assertEqual(len(ok["accepted_not_applicable"]), 1)
    moved = dataclasses.replace(
        run.expected_na, cells={("L2", "M4", 2, "indexer_agreement"): {**geometry, "max_reachable": 511}}
    )
    run_moved = mock.Mock(stage="L2", sample="M4", expected_na=moved)
    self.assertEqual(lw_stages.finalize_report(report(), run_moved)["status"], "FAIL")
    unlisted = mock.Mock(stage="L2", sample="S2048", expected_na=run.expected_na)
    self.assertEqual(lw_stages.finalize_report(report(), unlisted)["unaccepted_statuses"][0]["cell_listed"], False)
    for status in ("FAIL", "FAIL_FLOOR_LIMITED", "ILL_CONDITIONED_REF_GAP", "NOT_RUN_ON_BACKEND"):
      out = lw_stages.finalize_report(report({"label": "y", "status": status}), run)
      self.assertEqual((out["status"], out["status_counts"][status]), ("FAIL", 1), status)
    not_fired = lw_stages.finalize_report(report(controls={"NC_L2b": {"status": "DID_NOT_FIRE"}}), run)
    self.assertEqual(not_fired["status"], "FAIL")
    self.assertEqual(not_fired["negative_controls_not_fired"][0]["status"], "DID_NOT_FIRE")
    with self.assertRaisesRegex(ValueError, "outside"):
      lw_stages.finalize_report(report({"label": "z", "status": "SKIPPED"}), run)
    idx_pass = {
        "status": "PASS",
        "layers": {
            "layer_2": {
                "checks": [{"label": "layer_2.h_out", "status": "PASS"}],
                "agreement": {"indexer": {"status": "PASS"}},
            }
        },
        "negative_controls": {"NC_L2b": {"status": "FIRED"}},
    }
    with self.assertRaisesRegex(ValueError, "negative_controls must include NC_L2e"):
      lw_stages.finalize_report(idx_pass, run)

  def test_expected_not_applicable_file_is_strict_and_matches_its_own_geometry(self):
    pinned = lw_stages.load_expected_not_applicable(lw_stages.EXPECTED_NOT_APPLICABLE_PATH)
    yml_path = lw_checks.REPO_ROOT / "src/maxtext/configs/models/deepseek4.1-flash.yml"
    yml = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    compress_ratios = list(yml["compress_ratios"])
    seqs = {"M4": 1024, "S2048": 2048, "L2x": 408}
    for (stage, sample, layer, check), geometry in pinned.cells.items():
      positions = np.arange(seqs[sample])[None]
      ratio = int(compress_ratios[layer])
      if check == lw_l2.CHECK_INDEXER_AGREEMENT:
        recomputed = lw_routing.indexer_geometry(positions, ratio, int(yml["indexer_topk"]))
      else:
        recomputed = lw_routing.candidate_geometry(
            positions,
            ratio,
            int(yml["candidate_topk_blocks"]),
            int(yml["candidate_block_size"]),
        )
      self.assertEqual(recomputed, geometry, (stage, sample, layer, check))
    data = json.loads(lw_stages.EXPECTED_NOT_APPLICABLE_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
      path = pathlib.Path(tmp) / "na.json"
      path.write_text(json.dumps({**data, "entries": data["entries"] + data["entries"][:1]}), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "twice"):
        lw_stages.load_expected_not_applicable(path)

  def test_scan_geometry_parse(self):
    geometry = lw_stages.parse_geometry("2|3;4|5")
    self.assertEqual(geometry, lw_programs.ScanGeometry(prefix=(2,), repeats=((3,), (4,)), suffix=(5,)))


class LoaderTest(unittest.TestCase):

  def test_streamed_layer_weights_match_reference_and_unfilled_leaf_fails(self):
    fx = _MiniFixture.get()
    parts = fx.load_layer(fx.config, fx.mesh, 6)
    ref_layer = fx.ref.layers[6]
    np.testing.assert_array_equal(
        np.asarray(parts.weights["self_attention"]["wq_a"]["kernel"][...]), ref_layer.attn.wq_a.weight.detach().numpy().T
    )
    np.testing.assert_array_equal(
        np.asarray(parts.weights["mlp"]["MoeBlock_0"]["gate"]["bias_vl"][...]),
        ref_layer.ffn.gate.bias_vl.detach().numpy(),
    )
    table = np.asarray(parts.weights["engram"]["embed"][...])
    ref_table = ref_layer.engram.embed.weight.detach().numpy()
    np.testing.assert_array_equal(table[: ref_table.shape[0]], ref_table)
    np.testing.assert_array_equal(table[ref_table.shape[0] :], 0.0)

    build = lw_programs.layer_build_fn(fx.config, fx.mesh, 6)
    abstract = nnx.to_pure_dict(lw_model.abstract_parts(build, fx.config, fx.mesh).weights)
    abstract["unmapped_leaf"] = jax.ShapeDtypeStruct((4,), jnp.float32)
    with self.assertRaisesRegex(Exception, "unmapped_leaf"):
      lw_model.load_weights({"decoder": {"layers_6": abstract}}, fx.config, fx.ckpt)

  def test_engram_rows_dequantize_like_the_canonical_hook(self):
    if not HAS_TORCH:
      raise unittest.SkipTest("PyTorch required for reference model")
    rng = np.random.default_rng(0)
    rows, cols = 64, 256
    weight = torch.tensor(rng.standard_normal((rows, cols)), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.0 ** rng.integers(-4, 4, (rows, cols // 32)), dtype=torch.float32).to(torch.float8_e8m0fnu)
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      save_file({"layers.1.engram.embed.weight": weight, "layers.1.engram.embed.scale": scale}, str(d / "m.safetensors"))
      index = {
          "weight_map": {"layers.1.engram.embed.weight": "m.safetensors", "layers.1.engram.embed.scale": "m.safetensors"}
      }
      (d / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
      (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}), encoding="utf-8")
      picked = np.array([3, 17, 40])
      got = lw_model.read_engram_rows(d, 1, picked)
    full = param_mapping.dequantize_block_scaled(
        (
            weight.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn),
            scale.view(torch.uint8).numpy().view(ml_dtypes.float8_e8m0fnu),
        )
    )
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(full, np.float32)[picked])

  def test_load_full_engram_table_and_load_layer_without_engram_slice(self):
    fx = _MiniFixture.get()
    rng = np.random.default_rng(23)
    stored_rows, padded_rows, cols = 60, 64, 256
    weight = torch.tensor(rng.standard_normal((stored_rows, cols)), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor(2.0 ** rng.integers(-4, 4, (stored_rows, cols // 32)), dtype=torch.float32).to(
        torch.float8_e8m0fnu
    )
    expected = np.asarray(
        param_mapping.dequantize_block_scaled(
            (
                weight.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn),
                scale.view(torch.uint8).numpy().view(ml_dtypes.float8_e8m0fnu),
            )
        ),
        dtype=ml_dtypes.bfloat16,
    )
    with tempfile.TemporaryDirectory() as tmp:
      d = pathlib.Path(tmp)
      save_file({"layers.1.engram.embed.weight": weight, "layers.1.engram.embed.scale": scale}, str(d / "m.safetensors"))
      index = {
          "weight_map": {
              "layers.1.engram.embed.weight": "m.safetensors",
              "layers.1.engram.embed.scale": "m.safetensors",
          }
      }
      (d / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
      (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}), encoding="utf-8")
      for spec in (jax.sharding.PartitionSpec("fsdp", None), jax.sharding.PartitionSpec()):
        sharding = jax.sharding.NamedSharding(fx.mesh, spec)
        abstract = jax.ShapeDtypeStruct((padded_rows, cols), jnp.bfloat16, sharding=sharding)
        loaded = lw_model.load_full_engram_table(d, 1, abstract)
        self.assertEqual(loaded.shape, (padded_rows, cols))
        self.assertEqual(loaded.dtype, jnp.bfloat16)
        self.assertEqual(loaded.sharding, sharding)
        arr = np.asarray(loaded)
        np.testing.assert_array_equal(arr[:stored_rows], expected)
        np.testing.assert_array_equal(arr[stored_rows:], np.zeros((padded_rows - stored_rows, cols), ml_dtypes.bfloat16))

    unq_weight = torch.tensor(rng.standard_normal((stored_rows, cols)), dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmp_unq:
      d_unq = pathlib.Path(tmp_unq)
      save_file({"layers.1.engram.embed.weight": unq_weight}, str(d_unq / "m.safetensors"))
      (d_unq / "model.safetensors.index.json").write_text(
          json.dumps({"weight_map": {"layers.1.engram.embed.weight": "m.safetensors"}}), encoding="utf-8"
      )
      (d_unq / "config.json").write_text(json.dumps({}), encoding="utf-8")
      sharding = jax.sharding.NamedSharding(fx.mesh, jax.sharding.PartitionSpec("fsdp", None))
      abstract_unq = jax.ShapeDtypeStruct((padded_rows, cols), jnp.float32, sharding=sharding)
      loaded_unq = np.asarray(lw_model.load_full_engram_table(d_unq, 1, abstract_unq))
      np.testing.assert_array_equal(loaded_unq[:stored_rows], unq_weight.to(torch.float32).numpy())
      np.testing.assert_array_equal(loaded_unq[stored_rows:], np.zeros((padded_rows - stored_rows, cols), np.float32))

    parts_full, plan_none = lw_stages.load_layer(fx.config, fx.mesh, 1, fx.ckpt, engram=None)
    self.assertIsNone(plan_none)
    ref_parts = fx.load_layer(fx.config, fx.mesh, 1)
    np.testing.assert_array_equal(
        np.asarray(parts_full.weights["engram"]["embed"][...]),
        np.asarray(ref_parts.weights["engram"]["embed"][...]),
    )

  def test_engram_slice_matches_full_table_and_missing_row_poisons_output(self):
    fx = _MiniFixture.get()
    parts = fx.load_layer(fx.config, fx.mesh, 1)
    layer = parts.merge()
    io = (jnp.asarray(fx.positions), jnp.asarray(fx.tokens))
    carry = {"h": jnp.asarray(fx.h), "pre_mix": jnp.asarray(fx.pre_mix)}
    full_out = layer(dict(carry), *io, image_mask=jnp.asarray(fx.image_mask))
    hasher_kwargs = {"input_ids": jnp.asarray(fx.tokens), "image_mask": jnp.asarray(fx.image_mask)}
    active = np.unique(np.asarray(layer.engram_hasher.val(**hasher_kwargs)))
    num_rows = getattr(layer.engram, "num_rows", layer.engram.embed[...].shape[0])
    controls = np.setdiff1d(np.arange(num_rows), active)[:8]
    table = np.asarray(parts.weights["engram"]["embed"][...])
    table_sharding = parts.weights["engram"]["embed"][...].sharding

    def run_with(plan):
      lw_model.verify_slice_covers(layer.engram_hasher.val, plan, [hasher_kwargs])
      rows = table[plan.row_ids]
      sliced = jax.tree.map(lambda x: x, parts.weights)
      sliced["engram"]["embed"].set_value(lw_model.build_engram_slice_table(rows, plan, jnp.float32, table_sharding))
      sliced_layer = nnx.merge(parts.graphdef, sliced, parts.rest)
      lw_model.install_sliced_hasher(sliced_layer, plan)
      return sliced_layer(dict(carry), *io, image_mask=jnp.asarray(fx.image_mask))

    plan = lw_model.plan_engram_slice(active, controls, num_table_rows=num_rows, row_multiple=_NUM_DEVICES)
    np.testing.assert_array_equal(np.asarray(run_with(plan)["h"]), np.asarray(full_out["h"]))

    short = lw_model.plan_engram_slice(active[1:], controls, num_table_rows=num_rows, row_multiple=_NUM_DEVICES)
    with self.assertRaisesRegex(ValueError, "misses 1 reached rows"):
      lw_model.verify_slice_covers(layer.engram_hasher.val, short, [hasher_kwargs])
    rows = table[short.row_ids]
    sliced = jax.tree.map(lambda x: x, parts.weights)
    sliced["engram"]["embed"].set_value(lw_model.build_engram_slice_table(rows, short, jnp.float32, table_sharding))
    sliced_layer = nnx.merge(parts.graphdef, sliced, parts.rest)
    lw_model.install_sliced_hasher(sliced_layer, short)
    poisoned = np.asarray(sliced_layer(dict(carry), *io, image_mask=jnp.asarray(fx.image_mask))["h"])
    self.assertTrue(np.isnan(poisoned).any())
    self.assertTrue(np.isfinite(poisoned).any())

  def test_stage_loader_streams_an_engram_layer_with_its_table_as_a_verified_slice(self):
    fx = _MiniFixture.get()
    full = fx.load_layer(fx.config, fx.mesh, 1)
    layer = full.merge()
    active = np.unique(
        np.asarray(layer.engram_hasher.val(input_ids=jnp.asarray(fx.tokens), image_mask=jnp.asarray(fx.image_mask)))
    )
    num_rows = getattr(layer.engram, "num_rows", layer.engram.embed[...].shape[0])
    controls = np.setdiff1d(np.arange(num_rows), active)[:8]
    rows = lw_stages.EngramRows(active, controls, fx.tokens, fx.image_mask)
    parts, plan = lw_stages.load_layer(fx.config, fx.mesh, 1, fx.ckpt, rows)
    table = np.asarray(parts.weights["engram"]["embed"][...])
    self.assertEqual(table.shape[0], plan.num_slice_rows)
    np.testing.assert_array_equal(
        table[: plan.row_ids.size], np.asarray(full.weights["engram"]["embed"][...])[plan.row_ids]
    )
    self.assertTrue(np.isnan(table[plan.poison_row]).all())
    np.testing.assert_array_equal(
        np.asarray(parts.weights["engram"]["wkv"]["kernel"][...]),
        np.asarray(full.weights["engram"]["wkv"]["kernel"][...]),
    )
    short = lw_stages.EngramRows(active[1:], controls, fx.tokens, fx.image_mask, include_packing_rows=False)
    with self.assertRaisesRegex(ValueError, "misses 1 reached rows"):
      lw_stages.load_layer(fx.config, fx.mesh, 1, fx.ckpt, short)

  def test_engram_packing_controls_fire_with_nan_poison_row(self):
    # pylint: disable=protected-access
    fx = _MiniFixture.get()
    full = fx.load_layer(fx.config, fx.mesh, 1)
    layer = full.merge()
    active = np.unique(
        np.asarray(layer.engram_hasher.val(input_ids=jnp.asarray(fx.tokens), image_mask=jnp.asarray(fx.image_mask)))
    )
    num_rows = getattr(layer.engram, "num_rows", layer.engram.embed[...].shape[0])
    controls = np.setdiff1d(np.arange(num_rows), active)[:8]
    rows = lw_stages.EngramRows(active, controls, fx.tokens, fx.image_mask, include_packing_rows=True)
    parts, plan = lw_stages.load_layer(fx.config, fx.mesh, 1, fx.ckpt, rows)
    self.assertTrue(np.isnan(parts.weights["engram"]["embed"][...][plan.poison_row]).all())

    sample_inputs = lw_l2.SampleInputs(
        tokens=fx.tokens,
        positions=fx.positions,
        image_mask=fx.image_mask,
        text_pos_mask=~fx.image_mask,
        image_pos_mask=fx.image_mask,
    )
    carry = {"h": jnp.asarray(fx.h), "pre_mix": jnp.asarray(fx.pre_mix)}
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (1,), (parts.graphdef,), (plan,))
    replays = [{"moe_topk": jnp.zeros((_BATCH, fx.tokens.shape[1], fx.config.num_experts_per_tok), jnp.int32)}]
    l2_pack = lw_l2._packing_control(program, [parts.weights], [parts.rest], carry, sample_inputs, replays, 0)
    self.assertEqual(l2_pack["status"], "FIRED")
    self.assertEqual(l2_pack["max_abs_diff_clean"], 0.0)
    self.assertGreater(l2_pack["max_abs_diff_ones"], 1e-3)
    self.assertTrue(np.isfinite(l2_pack["max_abs_diff_ones"]))

    vjp_entry = jax.jit(lw_programs.layer_vjp_entry(program))
    cotangents = {"h": jnp.ones_like(carry["h"]), "pre_mix": jnp.ones_like(carry["pre_mix"])}
    l4_pack = lw_l4._packing_control(
        vjp_entry, parts.weights, parts.rest, carry, sample_inputs, replays[0], cotangents, 1
    )
    self.assertEqual(l4_pack["status"], "FIRED")
    self.assertEqual(l4_pack["grad_frob_clean"], 0.0)
    self.assertGreater(l4_pack["grad_frob_ones"], 1e-3)
    self.assertTrue(np.isfinite(l4_pack["grad_frob_clean"]))
    self.assertTrue(np.isfinite(l4_pack["grad_frob_ones"]))

  def test_chunked_sliding_window_packing_control_fires_at_s1024(self):
    # pylint: disable=protected-access
    fx = _MiniFixture.get()
    full = fx.load_layer(fx.config, fx.mesh, 1)
    layer = full.merge()

    b, s = 1, 1024
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 1000, size=(b, s), dtype=np.int32)
    img_mask = np.zeros((b, s), dtype=bool)
    positions = np.broadcast_to(np.arange(s, dtype=np.int32), (b, s))

    active = np.unique(
        np.asarray(layer.engram_hasher.val(input_ids=jnp.asarray(tokens), image_mask=jnp.asarray(img_mask)))
    )
    num_rows = getattr(layer.engram, "num_rows", layer.engram.embed[...].shape[0])
    controls = np.setdiff1d(np.arange(num_rows), active)[:8]
    rows = lw_stages.EngramRows(active, controls, tokens, img_mask, include_packing_rows=True)
    parts, plan = lw_stages.load_layer(fx.config, fx.mesh, 1, fx.ckpt, rows)

    sample_inputs = lw_l2.SampleInputs(
        tokens=tokens,
        positions=positions,
        image_mask=img_mask,
        text_pos_mask=~img_mask,
        image_pos_mask=img_mask,
    )

    merged = jnp.ones((b, s, fx.config.emb_dim), dtype=jnp.float32)
    carry = deepseek41.initial_carry(merged, fx.config.mhc_expansion_rate)

    program = lw_programs.GroupProgram(fx.config, fx.mesh, (1,), (parts.graphdef,), (plan,))
    replays = [{"moe_topk": jnp.zeros((b, s, fx.config.num_experts_per_tok), jnp.int32)}]

    l2_pack = lw_l2._packing_control(program, [parts.weights], [parts.rest], carry, sample_inputs, replays, 0)
    self.assertEqual(l2_pack["status"], "FIRED")
    self.assertEqual(l2_pack["max_abs_diff_clean"], 0.0)
    self.assertGreater(l2_pack["max_abs_diff_ones"], 1e-3)

    vjp_entry = jax.jit(lw_programs.layer_vjp_entry(program))
    cotangents = {"h": jnp.ones_like(carry["h"]), "pre_mix": jnp.ones_like(carry["pre_mix"])}
    l4_pack = lw_l4._packing_control(
        vjp_entry, parts.weights, parts.rest, carry, sample_inputs, replays[0], cotangents, 1
    )
    self.assertEqual(l4_pack["status"], "FIRED")
    self.assertEqual(l4_pack["grad_frob_clean"], 0.0)
    self.assertGreater(l4_pack["grad_frob_ones"], 1e-3)

    # Negative control demonstration of chunked sliding-window causal mask arithmetic:
    # With s_a = 513 (Doc A [0..512], Doc B [513..768]), Chunk 1 query [512..1023] slices keys
    # from [512 - 127 : 1024] = [385..1023]. The only Doc A keys in Chunk 1 are [385..512] (pos_k in 385..512).
    # All Doc B queries have reset pos_q in 0..255, giving delta = pos_q - pos_k <= 255 - 385 = -130 < 0,
    # so (delta >= 0) & (delta < 128) is identically False for all Doc A keys under seg_ones.
    chunk_size, window_size = 512, 128
    s_a_old, s_b_old = (s // 2) | 1, s // 4
    pos_q_old = np.arange(s_b_old)
    pos_k_old = np.arange(chunk_size - (window_size - 1), s_a_old)
    delta_old = pos_q_old[:, None] - pos_k_old[None, :]
    chunk1_mask_old = (delta_old >= 0) & (delta_old < window_size)
    self.assertEqual(chunk1_mask_old.sum(), 0)

    # With s_a = 65, s_b = 256, Doc A [0..64] and Doc B [65..320] both lie inside Chunk 0 [0..511],
    # where delta = pos_q - pos_k >= 0 & < 128 is populated under seg_ones.
    s_a_new, s_b_new = lw_model.packing_control_bounds(s)
    self.assertLessEqual(s_a_new + s_b_new, chunk_size)
    pos_q_new = np.arange(s_b_new)
    pos_k_new = np.arange(s_a_new)
    delta_new = pos_q_new[:, None] - pos_k_new[None, :]
    chunk0_mask_new = (delta_new >= 0) & (delta_new < window_size)
    self.assertGreater(chunk0_mask_new.sum(), 0)

  def test_lw_chain_multimodal_embedding_merge(self):
    fx = _MiniFixture.get()
    b, s = fx.tokens.shape[0], fx.tokens.shape[1]
    emb_dim = fx.config.emb_dim
    num_images = 2
    tokens_per_image = 4

    text_emb = jnp.ones((b, s, emb_dim), dtype=jnp.bfloat16)
    img_emb = jnp.full((b, num_images, tokens_per_image, emb_dim), 2.0, dtype=jnp.bfloat16)
    img_mask = np.zeros((b, s), dtype=bool)
    img_mask[:, : num_images * tokens_per_image] = True

    mock_l1_mod = mock.MagicMock()
    mock_l1_mod.vision_encoder.return_value = (img_emb, None)
    mock_l1_mod.token_embedder.return_value = text_emb

    mock_l1_parts = mock.MagicMock()
    mock_l1_parts.weights = {}
    mock_l1_parts.rest = {}
    mock_l1_parts.graphdef = None

    mock_bundle = mock.MagicMock()
    mock_sample_geom = mock.MagicMock()
    mock_sample_geom.num_images = num_images
    mock_sample_geom.seq_len = s
    mock_bundle.sample.return_value = mock_sample_geom
    mock_l1_ref = {
        "tokens": np.ones((b, s), dtype=np.int32),
        "pixel_values": np.zeros((b, num_images, 3, 16, 16), dtype=np.float32),
        "image_token_mask": img_mask,
    }
    mock_bundle.open.return_value = mock_l1_ref

    with (
        mock.patch.object(lw_stages, "load_l1", return_value=mock_l1_parts),
        mock.patch.object(lw_model, "merge_fresh", return_value=mock_l1_mod),
        mock.patch.object(lw_model, "release", return_value={"released_gib": 0.0}),
        mock.patch.object(deepseek41, "build_layer_specs", return_value=()),
    ):
      rep_fwd = lw_chain.lw_chain_forward(fx.config, fx.mesh, pathlib.Path("/tmp/mock_hf"), mock_bundle, "M4")
      self.assertEqual(rep_fwd.status, lw_checks.STATUS_PASS)
      self.assertTrue(np.isfinite(rep_fwd.metrics["final_h_norm"]))

      with tempfile.TemporaryDirectory() as tmp_dir:
        batch_path = pathlib.Path(tmp_dir) / "pinned_batch.npz"
        np.savez(
            batch_path,
            inputs=np.ones((b, s), dtype=np.int32),
            inputs_position=np.arange(s, dtype=np.int32)[None, :],
            pixel_values=np.zeros((b, num_images, 3, 16, 16), dtype=np.float32),
            image_token_mask=img_mask,
        )
        rep_pre = lw_chain.lw_chain_hidden_norm_consistency(fx.config, fx.mesh, pathlib.Path("/tmp/mock_hf"), batch_path)
        self.assertEqual(rep_pre["status"], lw_checks.STATUS_PASS)
        self.assertTrue(np.isfinite(rep_pre["total_consistency_objective"]))

  def test_lw_chain_hidden_norm_consistency_slices_engram_layers(self):
    """Verifies lw_chain_hidden_norm_consistency constructs EngramRows slices for engram layers, preventing OOM."""
    fx = _MiniFixture.get()
    num_images = 1
    b, s = 2, fx.config.max_target_length
    img_mask = np.zeros((b, s), dtype=bool)
    img_mask[:, :16] = True

    mock_l1_parts = mock.MagicMock()
    mock_l1_parts.weights = {}
    mock_l1_parts.rest = {}
    mock_l1_parts.graphdef = None

    mock_l1_mod = mock.MagicMock()
    mock_l1_mod.vision_encoder.return_value = (
        jnp.zeros((b, num_images, 16, fx.config.emb_dim), dtype=jnp.float32),
        None,
    )
    mock_l1_mod.token_embedder.return_value = jnp.zeros((b, s, fx.config.emb_dim), dtype=jnp.float32)

    spec_no_engram = deepseek41.LayerSpec(
        layer_id=0,
        compress_ratio=0,
        is_kv_source=False,
        is_index_source=False,
        is_candidate_source=False,
        uses_candidates=False,
        has_engram=False,
    )
    spec_with_engram = deepseek41.LayerSpec(
        layer_id=1,
        compress_ratio=0,
        is_kv_source=False,
        is_index_source=False,
        is_candidate_source=False,
        uses_candidates=False,
        has_engram=True,
    )

    loaded_engram_args = {}

    def fake_load_layer(cfg, msh, layer_idx, hf_path, engram, *args, **kwargs):
      del cfg, msh, hf_path, args, kwargs
      loaded_engram_args[layer_idx] = engram
      parts = mock.MagicMock()
      parts.weights = {}
      parts.rest = {}
      parts.graphdef = None
      plan = mock.MagicMock() if engram is not None else None
      return parts, plan

    mock_layer = mock.MagicMock()
    mock_layer.side_effect = lambda state, *args, **kwargs: state

    with (
        mock.patch.object(lw_stages, "load_l1", return_value=mock_l1_parts),
        mock.patch.object(lw_model, "merge_fresh", side_effect=[mock_l1_mod, mock_layer, mock_layer]),
        mock.patch.object(lw_model, "install_sliced_hasher") as mock_install_hasher,
        mock.patch.object(lw_model, "release", return_value={"released_gib": 0.0}),
        mock.patch.object(deepseek41, "build_layer_specs", return_value=[spec_no_engram, spec_with_engram]),
        mock.patch.object(lw_stages, "load_layer", side_effect=fake_load_layer),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
      batch_path = pathlib.Path(tmp_dir) / "pinned_batch.npz"
      np.savez(
          batch_path,
          inputs=np.ones((b, s), dtype=np.int32),
          inputs_position=np.arange(s, dtype=np.int32)[None, :],
          inputs_segmentation=np.ones((b, s), dtype=np.int32),
          pixel_values=np.zeros((b, num_images, 3, 16, 16), dtype=np.float32),
          image_token_mask=img_mask,
      )
      rep = lw_chain.lw_chain_hidden_norm_consistency(fx.config, fx.mesh, pathlib.Path("/tmp/mock_hf"), batch_path)
      self.assertEqual(rep["status"], lw_checks.STATUS_PASS)

      self.assertIsNone(loaded_engram_args[0])

      engram_arg = loaded_engram_args[1]
      self.assertIsInstance(engram_arg, lw_stages.EngramRows)
      self.assertGreater(len(engram_arg.active), 0)
      self.assertFalse(engram_arg.include_packing_rows)

      mock_install_hasher.assert_called_once()

  def test_lw_chain_step1_r4_parity_matches_monolithic_mini_config(self):
    """Verifies on 4-layer mini-config that lw_chain_step1_r4_parity matches monolithic forward+backward <= 1e-5."""
    fx = _MiniFixture.get()
    cfg = _mini_config(load_balance_loss_weight=0.0001)
    mesh = fx.mesh
    specs = deepseek41.build_layer_specs(cfg)[:4]
    num_layers = len(specs)

    tokens = jnp.asarray(fx.tokens)
    positions = jnp.asarray(fx.positions)
    image_mask = jnp.asarray(fx.image_mask)
    targets = jnp.roll(tokens, -1, axis=1)
    targets_seg = jnp.where(image_mask, 0, 1)
    total_weights = int(jnp.sum(targets_seg != 0))
    patches = mm_testing.torch_to_jax(fx.batch.patches_bnpd, jnp.float32)

    l1_parts = lw_stages.load_l1(cfg, mesh, fx.ckpt)
    g_l1, w_l1, r_l1 = l1_parts.graphdef, l1_parts.weights, l1_parts.rest

    layer_parts = [fx.load_layer(cfg, mesh, spec.layer_id) for spec in specs]
    w_layers = [p.weights for p in layer_parts]
    g_layers = [p.graphdef for p in layer_parts]
    r_layers = [p.rest for p in layer_parts]

    head = lw_chain.HeadModules(cfg, mesh, rngs=nnx.Rngs(params=0))
    g_head, w_head, r_head = nnx.split(head, nnx.Param, ...)

    def monolithic_step(w_l1_in, w_l_list, w_h, b_in, in_tot_w):
      in_tokens = b_in["tokens"]
      in_positions = b_in["positions"]
      in_targets = b_in["targets"]
      in_targets_seg = b_in["targets_seg"]
      in_image_mask = b_in["image_mask"]
      in_pixel_values = b_in.get("pixel_values")
      l1_mod = nnx.merge(g_l1, w_l1_in, r_l1)
      if in_pixel_values is not None:
        img_emb, _ = l1_mod.vision_encoder(input_images=in_pixel_values, deterministic=True)
        text_emb = l1_mod.token_embedder(in_tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
        merged = mm_utils.merge_mm_embeddings(
            text_embeddings=text_emb, multimodal_embeddings=img_emb, mask=in_image_mask, token_masks=None
        )
      else:
        merged = l1_mod.token_embedder(in_tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
      state = deepseek41.initial_carry(merged, cfg.mhc_expansion_rate)
      lb_losses = []
      for i in range(num_layers):
        mod = nnx.merge(g_layers[i], w_l_list[i], r_layers[i])
        state = mod(state, in_positions, in_tokens, decoder_segment_ids=in_targets_seg, image_mask=in_image_mask)
        inter = nnx.pop(mod, nnx.Intermediate)
        inter_dict = inter.to_pure_dict()
        lb_list = maxtext_utils.collect_intermediates_by_suffix(inter_dict, "moe_lb_loss")
        if lb_list:
          lb_losses.append(jnp.mean(jnp.concatenate(lb_list)))

      head_mod = nnx.merge(g_head, w_h, r_head)
      h_final = state["h"]
      if h_final.ndim == 4:
        h_final = jnp.sum(h_final, axis=2)
      logits = head_mod.logits_dense(head_mod.decoder_norm(h_final))
      one_hot = jax.nn.one_hot(in_targets, cfg.vocab_size)
      xent = optax.softmax_cross_entropy(logits, one_hot)
      lm_loss = jnp.sum(xent * (in_targets_seg != 0)) / float(in_tot_w)
      moe_lb_loss = jnp.mean(jnp.stack(lb_losses)) if lb_losses else jnp.asarray(0.0, dtype=jnp.float32)
      total_loss = lm_loss + moe_lb_loss
      return total_loss, (lm_loss, moe_lb_loss)

    batch_in = {
        "tokens": tokens,
        "positions": positions,
        "targets": targets,
        "targets_seg": targets_seg,
        "image_mask": image_mask,
        "pixel_values": patches,
    }
    (_, (mono_lm_loss, mono_moe_lb_loss)), (mono_d_l1, mono_d_layers, mono_d_head) = jax.value_and_grad(
        monolithic_step, argnums=(0, 1, 2), has_aux=True
    )(w_l1, w_layers, w_head, batch_in, total_weights)

    mono_grad_sq = 0.0
    for leaf in jax.tree.leaves(mono_d_head):
      mono_grad_sq += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
    for l_w in mono_d_layers:
      for leaf in jax.tree.leaves(l_w):
        mono_grad_sq += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
    for leaf in jax.tree.leaves(mono_d_l1):
      mono_grad_sq += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
    mono_grad_norm = float(np.sqrt(mono_grad_sq))

    mock_targets_seg = np.asarray(targets_seg)

    floor = {"lm_loss": 0.0, "moe_lb_loss": 0.0, "raw_grad_norm": 0.0}
    orig_load_l1 = lw_stages.load_l1

    with (
        mock.patch.object(deepseek41, "build_layer_specs", return_value=specs),
        mock.patch.object(lw_stages, "load_l1", side_effect=lambda c, m, h: orig_load_l1(c, m, fx.ckpt)),
        mock.patch.object(
            lw_stages,
            "load_layer",
            side_effect=lambda c, m, i, hf, engram: (fx.load_layer(c, m, specs[i].layer_id), None),
        ),
        mock.patch.object(
            lw_chain,
            "load_head",
            side_effect=lambda c, m, h: lw_model.ModuleParts(
                *nnx.split(lw_chain.HeadModules(c, m, rngs=nnx.Rngs(params=0)), nnx.Param, ...)
            ),
        ),
    ):
      mock_batch = {
          "inputs": np.asarray(tokens),
          "inputs_position": np.asarray(positions),
          "inputs_segmentation": np.asarray(targets_seg),
          "targets": np.asarray(targets),
          "targets_segmentation": mock_targets_seg,
          "pixel_values": np.asarray(patches),
          "image_token_mask": np.asarray(image_mask),
      }
      res = lw_chain.lw_chain_step1_r4_parity(
          cfg,
          mesh,
          fx.ckpt,
          batch=mock_batch,
          b_micro=1,
          measured_floor=floor,
      )
      mono_total_loss = float(mono_lm_loss + mono_moe_lb_loss)
      self.assertAlmostEqual(res["metrics"]["lm_loss"], float(mono_lm_loss), places=5)
      self.assertAlmostEqual(res["metrics"]["moe_lb_loss"], float(mono_moe_lb_loss), places=5)
      self.assertAlmostEqual(res["metrics"]["total_loss"], mono_total_loss, places=5)
      self.assertAlmostEqual(res["metrics"]["raw_grad_norm"], mono_grad_norm, places=5)

      # B=4 verification across microbatches with b_micro=2
      tokens_4 = np.repeat(np.asarray(tokens), 2, axis=0)
      positions_4 = np.repeat(np.asarray(positions), 2, axis=0)
      targets_4 = np.repeat(np.asarray(targets), 2, axis=0)
      targets_seg_4 = np.repeat(mock_targets_seg, 2, axis=0)
      image_mask_4 = np.repeat(np.asarray(image_mask), 2, axis=0)
      patches_4 = np.repeat(np.asarray(patches), 2, axis=0)
      total_weights_4 = int(np.sum(targets_seg_4 != 0))

      batch_in_4 = {
          "tokens": tokens_4,
          "positions": positions_4,
          "targets": targets_4,
          "targets_seg": targets_seg_4,
          "image_mask": image_mask_4,
          "pixel_values": patches_4,
      }
      (_, (mono_lm_loss_4, mono_moe_lb_loss_4)), (mono_d_l1_4, mono_d_layers_4, mono_d_head_4) = jax.value_and_grad(
          monolithic_step, argnums=(0, 1, 2), has_aux=True
      )(w_l1, w_layers, w_head, batch_in_4, total_weights_4)

      mono_grad_sq_4 = 0.0
      for leaf in jax.tree.leaves(mono_d_head_4):
        mono_grad_sq_4 += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
      for l_w in mono_d_layers_4:
        for leaf in jax.tree.leaves(l_w):
          mono_grad_sq_4 += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
      for leaf in jax.tree.leaves(mono_d_l1_4):
        mono_grad_sq_4 += float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
      mono_grad_norm_4 = float(np.sqrt(mono_grad_sq_4))

      mock_batch_4 = {
          "inputs": tokens_4,
          "inputs_position": positions_4,
          "inputs_segmentation": targets_seg_4,
          "targets": targets_4,
          "targets_segmentation": targets_seg_4,
          "pixel_values": patches_4,
          "image_token_mask": image_mask_4,
      }
      res_4 = lw_chain.lw_chain_step1_r4_parity(
          cfg,
          mesh,
          fx.ckpt,
          batch=mock_batch_4,
          b_micro=2,
          measured_floor=floor,
      )
      mono_total_loss_4 = float(mono_lm_loss_4 + mono_moe_lb_loss_4)
      self.assertAlmostEqual(res_4["metrics"]["lm_loss"], float(mono_lm_loss_4), places=5)
      self.assertAlmostEqual(res_4["metrics"]["moe_lb_loss"], float(mono_moe_lb_loss_4), places=5)
      self.assertAlmostEqual(res_4["metrics"]["total_loss"], mono_total_loss_4, places=5)
      self.assertAlmostEqual(res_4["metrics"]["raw_grad_norm"], mono_grad_norm_4, places=5)

  def test_lw_chain_step1_r4_parity_gate_and_negative_control(self):
    """Verifies design:155 gate logic, measured_floor recording, and NC_L5_step1_perturb negative control."""
    fx = _MiniFixture.get()
    cfg = fx.config
    mesh = fx.mesh
    specs = deepseek41.build_layer_specs(cfg)[:2]

    tokens = jnp.asarray(fx.tokens)
    positions = jnp.asarray(fx.positions)
    image_mask = jnp.asarray(fx.image_mask)
    targets = jnp.roll(tokens, -1, axis=1)
    targets_seg = jnp.where(image_mask, 0, 1)
    initial_carry = {"h": jnp.asarray(fx.h), "pre_mix": jnp.asarray(fx.pre_mix)}

    mock_l1_parts = mock.MagicMock()
    mock_l1_parts.weights = {}
    mock_l1_parts.rest = {}
    mock_l1_parts.graphdef = None

    mock_targets_seg = np.asarray(targets_seg)

    mock_batch = {
        "inputs": np.asarray(tokens),
        "inputs_position": np.asarray(positions),
        "inputs_segmentation": np.asarray(targets_seg),
        "targets": np.asarray(targets),
        "targets_segmentation": mock_targets_seg,
    }

    mock_l1_mod2 = mock.MagicMock()
    mock_l1_mod2.vision_encoder.return_value = (jnp.zeros_like(tokens)[:, :, None, None], None)
    mock_l1_mod2.token_embedder.return_value = jnp.zeros_like(tokens)[:, :, None]
    orig_merge_fresh2 = lw_model.merge_fresh

    def fake_merge_fresh2(g, w, r):
      if g is None:
        return mock_l1_mod2
      return orig_merge_fresh2(g, w, r)

    with (
        mock.patch.object(deepseek41, "build_layer_specs", return_value=specs),
        mock.patch.object(lw_stages, "load_l1", return_value=mock_l1_parts),
        mock.patch.object(lw_model, "merge_fresh", side_effect=fake_merge_fresh2),
        mock.patch.object(deepseek41, "initial_carry", return_value=initial_carry),
        mock.patch.object(
            lw_stages,
            "load_layer",
            side_effect=lambda c, m, i, hf, engram: (fx.load_layer(c, m, specs[i].layer_id), None),
        ),
        mock.patch.object(
            lw_chain,
            "load_head",
            side_effect=lambda c, m, h: lw_model.ModuleParts(
                *nnx.split(lw_chain.HeadModules(c, m, rngs=nnx.Rngs(params=0)), nnx.Param, ...)
            ),
        ),
    ):
      floor = {"lm_loss": 0.005, "moe_lb_loss": 0.00001, "raw_grad_norm": 1.0}
      res_clean = lw_chain.lw_chain_step1_r4_parity(
          cfg, mesh, pathlib.Path("/tmp/mock_hf"), batch=mock_batch, measured_floor=floor
      )
      self.assertEqual(res_clean["measured_floor_method"], "abs_diff_v5p8_fsdp4_ep1_vs_fsdp1_ep4")
      self.assertEqual(res_clean["measured_floor"], floor)
      self.assertIn("batch_sha256", res_clean)
      self.assertEqual(res_clean["batch_row_selection_source"], "src/maxtext/trainers/pre_train/train.py:168-171")
      self.assertEqual(res_clean["grad_norm_scope_source"], "src/maxtext/trainers/pre_train/train.py:586,638,801")

      res_nc = lw_chain.lw_chain_step1_r4_parity(
          cfg,
          mesh,
          pathlib.Path("/tmp/mock_hf"),
          batch=mock_batch,
          measured_floor=floor,
          negative_control=True,
          nc_perturb_leaf="logits_dense",
      )
      self.assertTrue(res_nc["negative_control"]["active"])
      self.assertTrue(res_nc["negative_control"]["nc_fired"])
      self.assertTrue(res_nc["passed"])

  def test_run_chained_step1_loss_and_grad_retain_layer_grads_parity(self):
    """Verifies retain_layer_grads=False matches True for losses and grad norm while freeing layer_grads."""
    fx = _MiniFixture.get()
    cfg = fx.config
    mesh = fx.mesh
    specs = [dataclasses.replace(s, has_engram=False) for s in deepseek41.build_layer_specs(cfg)[:2]]

    tokens = jnp.asarray(fx.tokens)
    positions = jnp.asarray(fx.positions)
    image_mask = jnp.asarray(fx.image_mask)
    targets = jnp.roll(tokens, -1, axis=1)
    targets_seg = jnp.where(image_mask, 0, 1)
    initial_carry = {"h": jnp.asarray(fx.h), "pre_mix": jnp.asarray(fx.pre_mix)}

    mock_l1_parts = mock.MagicMock()
    mock_l1_parts.weights = {}
    mock_l1_parts.rest = {}
    mock_l1_parts.graphdef = None

    mock_l1_mod = mock.MagicMock()
    mock_l1_mod.vision_encoder.side_effect = lambda **kw: (jnp.zeros((kw.get("images", tokens).shape[0], 1, 1, 1)), None)
    mock_l1_mod.token_embedder.side_effect = lambda t, **kw: jnp.zeros(t.shape + (cfg.base_emb_dim,), dtype=jnp.float32)

    mock_targets_seg = np.asarray(targets_seg)
    orig_merge_fresh = lw_model.merge_fresh

    def fake_merge_fresh(g, w, r):
      if g is None:
        return mock_l1_mod
      return orig_merge_fresh(g, w, r)

    mock_batch = {
        "inputs": np.asarray(tokens),
        "inputs_position": np.asarray(positions),
        "inputs_segmentation": np.asarray(targets_seg),
        "targets": np.asarray(targets),
        "targets_segmentation": mock_targets_seg,
    }

    with (
        mock.patch.object(deepseek41, "build_layer_specs", return_value=specs),
        mock.patch.object(lw_stages, "load_l1", return_value=mock_l1_parts),
        mock.patch.object(lw_model, "merge_fresh", side_effect=fake_merge_fresh),
        mock.patch.object(
            deepseek41,
            "initial_carry",
            side_effect=lambda h, hc: {
                k: jnp.repeat(v, max(1, h.shape[0] // tokens.shape[0]), axis=0) for k, v in initial_carry.items()
            },
        ),
        mock.patch.object(
            lw_stages,
            "load_layer",
            side_effect=lambda c, m, i, hf, engram: (fx.load_layer(c, m, specs[i].layer_id), None),
        ) as mock_load_layer,
        mock.patch.object(
            lw_chain,
            "load_head",
            side_effect=lambda c, m, h: lw_model.ModuleParts(
                *nnx.split(lw_chain.HeadModules(c, m, rngs=nnx.Rngs(params=0)), nnx.Param, ...)
            ),
        ),
    ):
      skeletons = {}
      with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = pathlib.Path(tmp_dir) / "weight_cache"

        res_true = lw_chain.run_chained_step1_loss_and_grad(
            cfg,
            mesh,
            pathlib.Path("/tmp/mock_hf"),
            mock_batch,
            b_micro=1,
            retain_layer_grads=True,
            weight_cache_dir=cache_dir,
            skeletons=skeletons,
        )
        self.assertEqual(mock_load_layer.call_count, len(specs))

        # Verify disk cache files were created on first pass
        for spec in specs:
          c_file = cache_dir / f"layer_{spec.layer_id:02d}.pkl"
          self.assertTrue(c_file.is_file())
          self.assertGreater(c_file.stat().st_size, 0)

        # Second pass reusing cache files with retain_layer_grads=False
        mock_load_layer.reset_mock()
        res_false = lw_chain.run_chained_step1_loss_and_grad(
            cfg,
            mesh,
            pathlib.Path("/tmp/mock_hf"),
            mock_batch,
            b_micro=1,
            retain_layer_grads=False,
            weight_cache_dir=cache_dir,
            skeletons=skeletons,
        )
        self.assertEqual(mock_load_layer.call_count, 0)

        np.testing.assert_allclose(res_false["lm_loss"], res_true["lm_loss"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res_false["moe_lb_loss"], res_true["moe_lb_loss"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res_false["total_loss"], res_true["total_loss"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res_false["raw_grad_norm"], res_true["raw_grad_norm"], rtol=1e-6, atol=1e-6)

        self.assertTrue(len(res_true["layer_grads"]) > 0)
        self.assertEqual(res_false["layer_grads"], {})
        self.assertEqual(res_false["head_grads"], {})
        self.assertEqual(res_false["emb_grads"], {})
        self.assertEqual(res_false["vision_proj_grads"], {})

        self.assertEqual(len(res_false["per_layer_grad_norms"]), len(specs))
        for spec in specs:
          i = spec.layer_id
          self.assertIn(i, res_false["per_layer_grad_norms"])
          np.testing.assert_allclose(
              res_false["per_layer_grad_norms"][i],
              res_true["per_layer_grad_norms"][i],
              rtol=1e-6,
              atol=1e-6,
          )

  def test_run_full_step1_r4_parity_suite_mini_config(self):
    """End-to-end verification of run_full_step1_r4_parity_suite on mini fixture."""
    fx = _MiniFixture.get()
    cfg = _mini_config(load_balance_loss_weight=0.0001)
    cfg_ep = _mini_config(
        ici_fsdp_parallelism=1,
        ici_expert_parallelism=_NUM_DEVICES,
        load_balance_loss_weight=0.0001,
    )
    mesh = fx.mesh
    specs = deepseek41.build_layer_specs(cfg)[:2]

    tokens = fx.tokens
    positions = fx.positions
    image_mask = fx.image_mask
    targets = np.roll(tokens, -1, axis=1)
    targets_seg = np.where(image_mask, 0, 1).astype(np.int32)
    patches = mm_testing.torch_to_jax(fx.batch.patches_bnpd, jnp.float32)

    mock_batch = {
        "inputs": np.asarray(tokens),
        "inputs_position": np.asarray(positions),
        "inputs_segmentation": targets_seg,
        "targets": np.asarray(targets),
        "targets_segmentation": targets_seg,
        "pixel_values": np.asarray(patches),
        "image_token_mask": np.asarray(image_mask),
    }

    orig_load_l1 = lw_stages.load_l1
    with tempfile.TemporaryDirectory() as tmp_dir:
      cache_dir = pathlib.Path(tmp_dir) / "weight_cache"
      out_dir = pathlib.Path(tmp_dir) / "suite_out"

      mock_ref = {
          "lm_loss": 12.0,
          "moe_lb_loss": 0.0001,
          "raw_grad_norm": 4.0,
      }

      with (
          mock.patch.object(deepseek41, "build_layer_specs", return_value=specs),
          mock.patch.object(lw_stages, "load_l1", side_effect=lambda c, m, h: orig_load_l1(c, m, fx.ckpt)),
          mock.patch.object(
              lw_stages,
              "load_layer",
              side_effect=lambda c, m, i, hf, engram: (fx.load_layer(c, m, specs[i].layer_id), None),
          ),
          mock.patch.object(
              lw_chain,
              "load_head",
              side_effect=lambda c, m, h: lw_model.ModuleParts(
                  *nnx.split(lw_chain.HeadModules(c, m, rngs=nnx.Rngs(params=0)), nnx.Param, ...)
              ),
          ),
          mock.patch.object(lw_config, "build_config", return_value=cfg_ep),
          mock.patch.object(lw_config, "build_mesh", return_value=mesh),
          mock.patch.object(lw_chain, "TPU7X_R4_STEP1_REF", mock_ref),
      ):
        # Populate cache
        lw_chain.run_chained_step1_loss_and_grad(
            cfg, mesh, fx.ckpt, batch_np=mock_batch, b_micro=1, retain_layer_grads=False, weight_cache_dir=cache_dir
        )
        # Measure reference from cached pass
        res_pass1 = lw_chain.run_chained_step1_loss_and_grad(
            cfg, mesh, fx.ckpt, batch_np=mock_batch, b_micro=1, retain_layer_grads=False, weight_cache_dir=cache_dir
        )
        mock_ref["lm_loss"] = float(res_pass1["lm_loss"])
        mock_ref["moe_lb_loss"] = float(res_pass1["moe_lb_loss"])
        mock_ref["raw_grad_norm"] = float(res_pass1["raw_grad_norm"])

        layer0_path = cache_dir / "layer_00.pkl"
        self.assertTrue(layer0_path.is_file(), "layer_00.pkl missing from cache")
        hash_before = hashlib.sha256(layer0_path.read_bytes()).hexdigest()

        summary = lw_chain.run_full_step1_r4_parity_suite(
            cfg,
            jax.devices(),
            fx.ckpt,
            batch=mock_batch,
            out_dir=out_dir,
            b_micro=1,
            weight_cache_dir=cache_dir,
        )

        self.assertTrue(summary["passed"])
        self.assertTrue(summary["primary_passed"])
        self.assertTrue(summary["neg_control_triggered"])
        self.assertTrue((out_dir / "step1_r4_parity_summary.json").is_file())

        hash_after = hashlib.sha256(layer0_path.read_bytes()).hexdigest()
        self.assertEqual(hash_before, hash_after, "layer_00.pkl mutated by Negative Control in Pass 3!")

  def test_release_frees_device_buffers(self):
    x = jax.device_put(jnp.ones((256, 256)), jax.devices()[0])
    report = lw_model.release({"x": x})
    self.assertTrue(x.is_deleted())
    self.assertAlmostEqual(report["released_gib"], 256 * 256 * 4 / 2**30)
    self.assertFalse(any(a is x for a in jax.live_arrays()))

  def test_device_memory_requires_stats_on_non_cpu_backend(self):
    with mock.patch.object(jax, "default_backend", return_value="cpu"):
      self.assertEqual(lw_model.device_memory("p"), {"label": "p", "backend": "cpu", "available": False})
    dev_none = mock.Mock(memory_stats=lambda: None)
    with (
        mock.patch.object(jax, "default_backend", return_value="tpu"),
        mock.patch.object(jax, "local_devices", return_value=[dev_none]),
    ):
      with self.assertRaisesRegex(RuntimeError, "unavailable or incomplete on non-CPU backend 'tpu'"):
        lw_model.device_memory("p")
    dev_partial = mock.Mock(memory_stats=lambda: {"bytes_in_use": 1024})
    with (
        mock.patch.object(jax, "default_backend", return_value="tpu"),
        mock.patch.object(jax, "local_devices", return_value=[dev_partial]),
    ):
      with self.assertRaisesRegex(RuntimeError, "unavailable or incomplete on non-CPU backend 'tpu'"):
        lw_model.device_memory("p")
    dev_ok = mock.Mock(memory_stats=lambda: {"peak_bytes_in_use": 2**30, "bytes_in_use": 2**29})
    with (
        mock.patch.object(jax, "default_backend", return_value="tpu"),
        mock.patch.object(jax, "local_devices", return_value=[dev_ok]),
    ):
      self.assertEqual(
          lw_model.device_memory("p"),
          {
              "label": "p",
              "available": True,
              "peak_bytes_in_use": [2**30],
              "bytes_in_use": [2**29],
              "max_peak_gib": 1.0,
          },
      )


class L3InvarianceTest(unittest.TestCase):
  """L3 on the mini twin. The chain test runs op-for-op; compiled scan / remat variants are compared on TPU."""

  def test_scan_vs_unscan_and_remat_on_mini_layers_with_negative_controls(self):
    fx = _MiniFixture.get()
    geometry = lw_programs.ScanGeometry(prefix=(8,), repeats=((9, 10), (11, 12)), suffix=(13,))
    prefix_parts = [fx.load_layer(fx.config, fx.mesh, i) for i in range(8)]
    chain = lw_l3.load_chain(fx.config, fx.mesh, geometry.layer_ids, fx.load_layer)
    carry = fx.carry_for(fx.config, 8, prefix_parts)
    inputs = lw_l3.ChainInputs(
        carry={k: np.asarray(v) for k, v in carry.items()},
        positions=fx.positions,
        tokens=fx.tokens,
        image_mask=fx.image_mask,
        cotangent_h=np.random.default_rng(5).standard_normal(fx.h.shape).astype(np.float32),
    )
    # Op-for-op execution: the gate's own logic and the production scan / remat appliers are under test here, not
    # XLA's fusion choices, which differ between the compiled scan / remat variants.
    with jax.disable_jit():
      report = lw_l3.run_chain_invariance(
          chain,
          _mini_config(remat_policy="full"),
          _mini_config(scan_layers=True),
          geometry,
          inputs,
          _TOL,
          "mlp/shared_experts/wo/kernel",
      )
    for family in ("remat", "scan_vs_unscan"):
      self.assertEqual(report[family]["status"], "PASS", report[family])
      self.assertEqual(report[family]["negative_control"]["status"], "FIRED", report[family]["negative_control"])
    grad_checks = [c for c in report["scan_vs_unscan"]["checks"] if c["label"].startswith("scan.grad/")]
    self.assertEqual(
        len(grad_checks), len([c for c in report["remat"]["checks"] if c["label"].startswith("remat.grad/")])
    )
    self.assertTrue(any("/mlp/MoeBlock_0/wi_0" in c["label"] for c in grad_checks))

  def test_ep_and_moe_implementations_on_one_mini_layer(self):
    fx = _MiniFixture.get()
    prefix_parts = [fx.load_layer(fx.config, fx.mesh, i) for i in range(3)]
    base = fx.load_layer(fx.config, fx.mesh, 3)
    carry = fx.carry_for(fx.config, 3, prefix_parts)
    configs = {"dense_ep1": (fx.config, fx.mesh)}
    for name, overrides in (
        ("dense_natural", {}),
        ("dense_epn", {"ici_fsdp_parallelism": 1, "ici_expert_parallelism": _NUM_DEVICES}),
        ("sparse_megablox", {"sparse_matmul": True, "megablox": True}),
        ("sparse_ragged", {"sparse_matmul": True, "megablox": False}),
    ):
      cfg = _mini_config(**overrides)
      configs[name] = (cfg, lw_config.build_mesh(cfg, jax.devices()))
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (3,), (base.graphdef,), (None,))
    replay = _natural_replay(program, base, carry, fx)
    inputs = lw_l3.ChainInputs(
        carry={k: np.asarray(v) for k, v in carry.items()},
        positions=fx.positions,
        tokens=fx.tokens,
        image_mask=fx.image_mask,
        cotangent_h=np.zeros_like(fx.h),
    )
    report = lw_l3.run_moe_invariance(
        3, base, configs, inputs, {k: np.asarray(v) for k, v in replay.items()}, _TOL, "mlp/shared_experts/wo/kernel"
    )
    self.assertEqual(report["ep1_vs_epn"]["status"], "PASS", report["ep1_vs_epn"])
    self.assertEqual(report["ep1_vs_epn"]["expert_parallelism"], _NUM_DEVICES)
    self.assertEqual(report["ep1_vs_epn"]["negative_control"]["status"], "FIRED")
    impl = report["moe_implementations_vs_dense"]
    self.assertEqual(impl["sparse_ragged"]["status"], "PASS", impl["sparse_ragged"])
    self.assertEqual(impl["sparse_ragged"]["negative_control"]["status"], "FIRED")
    for family in (report["ep1_vs_epn"], impl["sparse_ragged"]):
      labels = {c["label"].split("/", 1)[1] for c in family["checks"]}
      self.assertTrue(
          {"forward/h", "forward/pre_mix", "grad/d_h_in", "grad/d_pre_mix_in", "grad/0/mlp/MoeBlock_0/wi_0"} <= labels
      )
    self.assertEqual(impl["sparse_megablox"]["status"], "NOT_RUN_ON_BACKEND")
    self.assertEqual(set(report["activation_quant"]), set(configs))


class Fp64ParityTest(unittest.TestCase):
  """jax_ml_parity §5 on the L1 path: the harness's L1 program and its VJP, upcast to fp64 on both sides, agree with
  the reference to rel-Frobenius <= 1e-10 on the merged output and on every parameter-leaf gradient; moving one
  vision leaf with negctl makes the same comparison fail.

  Upcasting uses the test-scoped patches `mm_testing.fp64_maxtext_context` and `multimodal.fp64_reference_context`.
  A decoder layer is not covered here: see the round-3 report (the reference `Block` and the MaxText layer hard-cast
  to float32 at many sites, e.g. deepseek_v41_reference/model.py:938/945 and deepseek41.py:236/266/269).
  """

  NC_LEAF = "vision_encoder/DeepSeek41VisionEncoder_0/blocks_0/attn/wo/kernel"

  def _fp64_l1(self, fx, config, ref64):
    parts = lw_model.abstract_parts(lw_programs.l1_build_fn(config, fx.mesh), config, fx.mesh)
    module = nnx.merge(parts.graphdef, jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), parts.weights), parts.rest)
    mm_testing.copy_full_vision_encoder(module.vision_encoder, ref64, jnp.float64)
    module.token_embedder.embedding[...] = mm_testing.torch_to_jax(ref64.embed.weight, jnp.float64)
    return nnx.split(module, lw_model.WEIGHT_FILTER, ...)

  def test_l1_program_fp64_frobenius_forward_and_every_leaf_gradient(self):
    fx = _MiniFixture.get()
    config = _mini_config(freeze_vision_encoder_params=False)
    with multimodal.fp64_reference_context(), mm_testing.fp64_maxtext_context(config):
      ref64 = copy.deepcopy(fx.ref).double()
      batch = multimodal.build_mini_vl_batch(multimodal.mini_model_args(), seed=101, dtype=torch.float64)
      ref_out = ref64.merge_image_embeddings(batch.images, ref64.embed(batch.input_ids))
      cot = np.random.default_rng(31).standard_normal(tuple(ref_out.shape))
      (ref_out * torch.from_numpy(cot)).sum().backward()
      ref_merged = ref_out.detach().numpy()
      ref_grads = {
          name: (np.zeros(tuple(t.shape)) if t.grad is None else t.grad.numpy()) for name, t in ref64.named_parameters()
      }
      graphdef, weights, rest = self._fp64_l1(fx, config, ref64)
      tokens = jnp.asarray(batch.input_ids.numpy(), jnp.int32)
      patches = mm_testing.torch_to_jax(batch.patches_bnpd, jnp.float64)
      forward = jax.jit(lw_programs.l1_forward_entry(graphdef, config, capture=False))
      outputs, _ = forward(weights, rest, tokens, patches)
      cand = np.asarray(outputs["merged"])
      d_w, _ = jax.jit(lw_programs.l1_vjp_entry(graphdef, config))(weights, rest, tokens, patches, jnp.asarray(cot))
      moved = lw_model.perturb_leaf(weights, self.NC_LEAF, _TOL.negative_control_rel_scale, seed=0)
      nc_cand = np.asarray(forward(moved, rest, tokens, patches)[0]["merged"])
    self.assertEqual(cand.dtype, np.float64)
    ok, metrics = layer_parity.compare(cand, ref_merged, layer_parity.FP64_TOLERANCE)
    self.assertTrue(ok, metrics)
    np.testing.assert_array_equal(np.asarray(outputs["image_mask"]), batch.image_mask.numpy())
    names, hooks = lw_l1.l1_hf_mapping(config)
    leaves = lw_l4.grads_by_path(d_w)
    self.assertGreater(len(leaves), 10)
    self.assertEqual(set(leaves), set(names))
    expected_l1_ref = {n for n in ref_grads if not n.startswith(("layers.", "norm.", "head."))}
    self.assertEqual({names[p] for p in leaves} | {"embed.weight"}, expected_l1_ref)
    for path, grad in sorted(leaves.items()):
      self.assertEqual(grad.dtype, np.float64, path)
      ref_grad = ref_grads[names[path]]
      cand_grad = lw_l4.to_hf(grad, hooks[path], ref_grad.shape)
      if not np.any(ref_grad):
        np.testing.assert_array_equal(cand_grad, np.zeros_like(cand_grad), err_msg=f"{path}: reference gradient is 0")
        continue
      leaf_ok, leaf_metrics = layer_parity.compare(cand_grad, ref_grad, layer_parity.FP64_TOLERANCE)
      self.assertTrue(leaf_ok, (path, leaf_metrics))
    nc_ok, nc_metrics = layer_parity.compare(nc_cand, ref_merged, layer_parity.FP64_TOLERANCE)
    self.assertFalse(nc_ok, nc_metrics)


class LocalizationTest(unittest.TestCase):

  def test_capture_records_production_intermediates_in_forward_order(self):
    fx = _MiniFixture.get()
    parts = fx.load_layer(fx.config, fx.mesh, 1)
    layer = parts.merge()
    with lw_programs.capture_calls(
        layer, lw_programs.DECODER_CAPTURE_POINTS, lw_programs.DECODER_STATIC_CAPTURE_POINTS
    ) as store:
      out = layer(
          {"h": jnp.asarray(fx.h), "pre_mix": jnp.asarray(fx.pre_mix)},
          jnp.asarray(fx.positions),
          jnp.asarray(fx.tokens),
          image_mask=jnp.asarray(fx.image_mask),
      )
    self.assertEqual(
        set(store),
        {
            "engram:__call__#0",
            ":hc_mixes#0",
            ":hc_mixes#1",
            "pre_self_attention_layer_norm:__call__#0",
            "post_self_attention_layer_norm:__call__#0",
            "self_attention:__call__#0",
            "self_attention/q_norm:__call__#0",
            "self_attention/kv_norm:__call__#0",
            "mlp:__call__#0",
            "mlp/MoeBlock_0:__call__#0",
            "mlp/MoeBlock_0/gate:__call__#0",
            "DeepSeek41DecoderLayer.hc_pre#0",
            "DeepSeek41DecoderLayer.hc_pre#1",
            "DeepSeek41DecoderLayer.hc_post#0",
            "DeepSeek41DecoderLayer.hc_post#1",
        },
    )
    np.testing.assert_array_equal(np.asarray(store["DeepSeek41DecoderLayer.hc_post#1"]), np.asarray(out["h"]))

  def test_default_precision_hc_post_is_same_math_with_lower_precision(self):
    self.assertEqual(lw_programs.hc_post_source_sha256(), lw_programs.ORIGINAL_HC_POST_SHA256)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 8, 64)), jnp.float32)
    residual = jnp.asarray(rng.standard_normal((1, 8, 4, 64)), jnp.float32)
    post = jnp.asarray(rng.standard_normal((1, 8, 4)), jnp.float32)
    comb = jnp.asarray(rng.standard_normal((1, 8, 4, 4)), jnp.float32)
    original = deepseek41.DeepSeek41DecoderLayer.hc_post
    hlo_original = jax.jit(original).lower(x, residual, post, comb).as_text()
    with lw_programs.hc_post_default_precision():
      patched = deepseek41.DeepSeek41DecoderLayer.hc_post
      hlo_patched = jax.jit(patched).lower(x, residual, post, comb).as_text()
      self.assertIs(patched, lw_programs.default_precision_hc_post)
    self.assertIn("HIGHEST", hlo_original)
    self.assertNotIn("HIGHEST", hlo_patched)
    if jax.default_backend() == "cpu":  # CPU ignores precision=, so identical math gives identical values here.
      np.testing.assert_array_equal(
          np.asarray(jax.jit(lw_programs.default_precision_hc_post)(x, residual, post, comb)),
          np.asarray(jax.jit(original)(x, residual, post, comb)),
      )

  @unittest.skipUnless(
      jax.default_backend() == "tpu",
      "CPU matmuls ignore precision=, so the DEFAULT-precision hc_post "
      "control cannot change any value on CPU; it fires only on the TPU MXU (v5p runner).",
  )
  def test_default_precision_hc_post_changes_values_on_tpu(self):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((1, 64, 512)), jnp.float32)
    residual = jnp.asarray(rng.standard_normal((1, 64, 4, 512)), jnp.float32)
    post = jnp.asarray(rng.standard_normal((1, 64, 4)), jnp.float32)
    comb = jnp.asarray(rng.standard_normal((1, 64, 4, 4)), jnp.float32)
    self.assertFalse(
        np.array_equal(
            np.asarray(jax.jit(lw_programs.default_precision_hc_post)(x, residual, post, comb)),
            np.asarray(jax.jit(deepseek41.DeepSeek41DecoderLayer.hc_post)(x, residual, post, comb)),
        )
    )


def _natural_group_replays(program, parts, carry, fx):
  """Replay dicts equal to each layer's own natural choice when the group runs on them (fixed point of replay ->
  natural; an earlier layer's routing feeds later layers, and an index source's mask feeds its own MoE input)."""
  seq = fx.config.max_target_length
  replays = []
  for layer in (p.merge() for p in parts):
    replay = {"moe_topk": jnp.zeros((_BATCH, seq, fx.config.num_experts_per_tok), jnp.int32)}
    if layer.spec.is_index_source:
      replay["indexer_mask"] = jnp.ones((_BATCH, seq, seq // layer.spec.compress_ratio), bool)
    replays.append(replay)
  fwd = jax.jit(lw_programs.group_forward_entry(program, capture=False))
  args = (
      [p.weights for p in parts],
      [p.rest for p in parts],
      carry,
      jnp.asarray(fx.positions),
      jnp.asarray(fx.tokens),
      jnp.asarray(fx.image_mask),
  )
  for _ in range(2 * len(parts) + 2):
    outs, naturals, _ = fwd(*args, replays)
    if all(all(np.array_equal(np.asarray(n[k]), np.asarray(r[k])) for k in r) for n, r in zip(naturals, replays)):
      return replays, outs
    replays = [dict(n) for n in naturals]
  raise AssertionError("Group replays did not reach their natural fixed point")


def _hf_shapes(ckpt: pathlib.Path) -> dict:
  with open(ckpt / "model.safetensors.index.json", "r", encoding="utf-8") as f:
    weight_map = json.load(f)["weight_map"]
  shapes = {}
  for shard in sorted(set(weight_map.values())):
    with safe_open(str(ckpt / shard), framework="pt") as sf:
      shapes.update({k: tuple(sf.get_slice(k).get_shape()) for k in sf.keys()})
  return shapes


def _tri_variant(name: str, value) -> dict:
  """A gated bundle tensor in both tiers: fp32 (`name`), fp64 (`_fp64`), and well-conditioned bf16 (`_bf16`)."""
  arr = np.asarray(value)
  fp32 = arr.astype(np.float32)
  flat = fp32.reshape(-1).astype(np.float64)
  norm = float(np.linalg.norm(flat))
  scale = (norm / np.sqrt(max(1, flat.size))) if norm > 0.0 else 1.0
  pattern = np.where(np.arange(flat.size) % 2 == 0, 1.0, -0.5).reshape(fp32.shape)
  bf16 = (fp32.astype(np.float64) + 1e-3 * scale * pattern).astype(np.float32)
  return {
      name: fp32,
      name + lw_bundle.FP64_SUFFIX: arr.astype(np.float64),
      name + lw_bundle.BF16_SUFFIX: bf16,
  }


def _mask_to_ids(mask) -> np.ndarray:
  """[B, S, C] bool selection -> [B, S, K] int32 ids padded with -1 (the capture's top-k id layout)."""
  mask = np.asarray(mask)
  width = max(1, int(mask.sum(-1).max()))
  ids = np.full(mask.shape[:2] + (width,), -1, np.int32)
  for b, s in np.ndindex(*mask.shape[:2]):
    picked = np.flatnonzero(mask[b, s])
    ids[b, s, : picked.size] = picked
  return ids


class _BundleWriter:
  """Writes .npz bundles under `root` and indexes them like bundle_index.json (sha256 + key shapes / dtypes)."""

  def __init__(self, root: pathlib.Path):
    self.root = root
    self.entries = {}

  def write(self, relative_path: str, arrays: dict) -> None:
    path = self.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    keys = {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)} for k, v in arrays.items()}
    self.entries[relative_path] = {"sha256": lw_checks.sha256_file(path), "keys": keys}

  def index(self) -> lw_bundle.BundleIndex:
    lines = [f"{e['sha256']}  {rel}" for rel, e in sorted(self.entries.items())]
    (self.root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lw_bundle.BundleIndex(root=self.root, sha256="synthetic-index", data={"bundles": self.entries})


class SyntheticBundleStageTest(unittest.TestCase):
  """`lw_stages.run_stage` end to end for L1 forward, L1 vjp, L2 and L4 on the mini twin, from an indexed bundle
  written with the reference key names the adapters read (lw_bundle). The reference values are the mini twin's own
  outputs, so what is under test is the wiring: bundle adapters, AOT matching, loaders, gates, negative controls and
  the run verdict (`finalize_report`), in tier a. MaxText configs come from the mini twin (test-scoped patch)."""

  SAMPLE = "mini"

  @classmethod
  def setUpClass(cls):
    cls.fx = _MiniFixture.get()
    cls.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    cls.writer = _BundleWriter(pathlib.Path(cls.tmp.name))
    cls.configs = {
        "forward": _mini_config(freeze_vision_encoder_params=True),
        "vjp": _mini_config(freeze_vision_encoder_params=False),
    }
    cls.rng = np.random.default_rng(23)
    cls._write_l1_bundles()
    cls._write_decoder_bundles()
    cls.index = cls.writer.index()

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  @classmethod
  def _write_l1_bundles(cls):
    """L1 forward and L1-VJP bundles from the mini twin's L1 program and its VJP."""
    fx = cls.fx
    patches = mm_testing.torch_to_jax(fx.batch.patches_bnpd, jnp.float32)
    tokens = jnp.asarray(fx.tokens, jnp.int32)
    fwd_parts = lw_stages.load_l1(cls.configs["forward"], fx.mesh, fx.ckpt)
    outputs, captured = jax.jit(lw_programs.l1_forward_entry(fwd_parts.graphdef, cls.configs["forward"], capture=True))(
        fwd_parts.weights, fwd_parts.rest, tokens, patches
    )
    patch_embed = np.asarray(captured[f"{lw_l1.ENCODER}:embed_patches#0"])
    n_blocks = cls.configs["forward"].num_hidden_layers_for_vit
    vit_norm = np.asarray(captured[f"{lw_l1.ENCODER}:__call__#0"]).reshape(patch_embed.shape[0], patch_embed.shape[1], -1)
    aq = np.asarray(bool(fx.config.ds41_activation_quant))
    cls.writer.write(
        lw_bundle.l1_path(cls.SAMPLE),
        {
            "tokens": np.asarray(tokens),
            "pixel_values": np.asarray(patches),
            "image_grid_hw": np.asarray(cls.configs["forward"].ds41_vision_grid, np.int32),
            "image_token_mask": np.asarray(outputs["image_mask"]),
            "reference_activation_quant_fp64": aq,
            "reference_activation_quant_fp32": aq,
            **_tri_variant("patch_embed_out", patch_embed),
            **_tri_variant(
                "vit_block_out",
                np.stack([np.asarray(captured[f"{lw_l1.ENCODER}/blocks_{i}:__call__#0"]) for i in range(n_blocks)]),
            ),
            **_tri_variant("vit_norm_out", vit_norm),
            **_tri_variant("aligner_out", np.asarray(captured[f"{lw_l1.PROJECTOR}:align#0"])),
            **_tri_variant("post_merge_embeddings", np.asarray(outputs["merged"])),
            **_tri_variant("h0", np.asarray(outputs["h0"])),
            **_tri_variant("pre_mix0", np.asarray(outputs["pre_mix0"])),
        },
    )
    config = cls.configs["vjp"]
    vjp_parts = lw_stages.load_l1(config, fx.mesh, fx.ckpt)
    cot = cls.rng.standard_normal(np.asarray(outputs["merged"]).shape).astype(np.float32)
    d_w, d_x = jax.jit(lw_programs.l1_vjp_entry(vjp_parts.graphdef, config))(
        vjp_parts.weights, vjp_parts.rest, tokens, patches, jnp.asarray(cot)
    )
    span, image_mask = outputs["span"], outputs["image_mask"]

    def merged_from_text(text):
      merged = lw_programs.mm_utils.merge_mm_embeddings(
          text_embeddings=text, multimodal_embeddings=span, mask=image_mask, token_masks=None
      )
      return jnp.sum(merged.astype(jnp.float32) * cot)

    text = vjp_parts.merge().token_embedder(tokens, model_mode=lw_programs.MODEL_MODE_TRAIN)
    d_text = jax.grad(merged_from_text)(text)
    names, hooks = lw_l1.l1_hf_mapping(config)
    hf_shapes = _hf_shapes(fx.ckpt)
    leaves = {}
    for path, grad in lw_l4.grads_by_path(d_w).items():
      if path == "token_embedder/embedding":
        continue
      leaves.update(_tri_variant(f"dW_{names[path]}", lw_l4.to_hf(grad, hooks[path], hf_shapes[names[path]])))
    cls.writer.write(
        lw_bundle.l1_vjp_path(cls.SAMPLE),
        {
            "tokens": np.asarray(tokens),
            "image_token_mask": np.asarray(outputs["image_mask"]),
            "cotangent_span": cot,
            **_tri_variant("d_images", np.asarray(d_x)),
            **_tri_variant("d_embed_tokens", np.asarray(d_text)),
            **leaves,
        },
    )

  @classmethod
  def _common(cls) -> dict:
    fx = cls.fx
    return {
        "tokens": fx.tokens,
        "positions": fx.positions,
        "image_token_mask": fx.image_mask,
        "text_pos_mask": ~fx.image_mask,
        "image_pos_mask": fx.image_mask,
    }

  @classmethod
  def _routing_keys(cls, prefix: str, replay: dict, *, include_margins: bool = True) -> dict:
    """Replayed routing plus (for L2) high margins (9 ULP of the boundary score) for every position."""
    ulp = np.full(np.asarray(replay["moe_topk"]).shape[:2], float(2.0**-24), np.float32)
    high_margin = np.full(ulp.shape, 9.0 * float(2.0**-8), np.float32)
    keys = {f"{prefix}ref_topk_experts": np.asarray(replay["moe_topk"], np.int32)}
    if include_margins:
      keys.update({f"{prefix}ref_gate_margins": high_margin, f"{prefix}ref_gate_boundary_ulp": ulp})
    if "indexer_mask" in replay:
      keys.update({f"{prefix}ref_indexer_topk": _mask_to_ids(replay["indexer_mask"])})
      if include_margins:
        keys.update({f"{prefix}ref_indexer_margins": high_margin, f"{prefix}ref_indexer_boundary_ulp": ulp})
    return keys

  @classmethod
  def _carry_keys(cls, prefix: str, layer_id: int, state: dict, *, h_key: str, pre_mix_key: str) -> dict:
    """The carry keys `lw_bundle.carry_in` reads for `layer_id` (shared state under its `*_in` names)."""
    fx = cls.fx
    specs = deepseek41.build_layer_specs(fx.config)
    keys = {h_key: np.asarray(state["h"], np.float32), pre_mix_key: np.asarray(state["pre_mix"], np.float32)}
    for read in lw_programs.layer_reads(fx.config, layer_id):
      value = np.asarray(state[read])
      if read == "compress_kv":
        keys[f"{prefix}shared_compress_kv_in"] = value.astype(np.float32)
      elif read == "index_k":
        keys[f"{prefix}shared_index_k_in"] = value.astype(np.float32)
      elif read == "topk_mask":
        keys[f"{prefix}topk_indices_in"] = _mask_to_ids(value)
        assert value.shape[-1] == fx.config.max_target_length // specs[layer_id].compress_ratio
      else:
        keys[f"{prefix}candidates_mask_in"] = value
    return keys

  @classmethod
  def _write_l2_group(cls, layer_ids: tuple, prefix_ids: tuple):
    """One L2 bundle per layer of the group, from the group run on its natural routing fixed point."""
    fx = cls.fx
    prefix = [fx.load_layer(fx.config, fx.mesh, i) for i in prefix_ids]
    parts = [fx.load_layer(fx.config, fx.mesh, i) for i in layer_ids]
    carry = fx.carry_for(fx.config, layer_ids[0], prefix)
    program = lw_programs.GroupProgram(
        fx.config, fx.mesh, layer_ids, tuple(p.graphdef for p in parts), (None,) * len(parts)
    )
    replays, _ = _natural_group_replays(program, parts, carry, fx)
    outs, _, captured = jax.jit(lw_programs.group_forward_entry(program, capture=True))(
        [p.weights for p in parts],
        [p.rest for p in parts],
        carry,
        jnp.asarray(fx.positions),
        jnp.asarray(fx.tokens),
        jnp.asarray(fx.image_mask),
        replays,
    )
    state = dict(carry)
    specs = deepseek41.build_layer_specs(fx.config)
    for i, (layer_id, replay, out) in enumerate(zip(layer_ids, replays, outs)):
      p = f"layer_{layer_id}_"
      arrays = {**cls._common(), **cls._carry_keys(p, layer_id, state, h_key=f"{p}h_in", pre_mix_key=f"{p}pre_mix_in")}
      arrays.update(_tri_variant(f"{p}h_out", out["h"]))
      arrays.update(_tri_variant(f"{p}pre_mix_out", out["pre_mix"]))
      for key in lw_programs.layer_writes(fx.config, layer_id):
        if key in lw_l2.GATED_SHARED_OUTPUTS:
          arrays.update(_tri_variant(f"{p}{key}_out", out[key]))
      for capture_key, idx, name in lw_l2.localization_points(specs[layer_id]):
        if name == "h_out":
          continue
        val = np.asarray(captured[i][capture_key] if idx is None else captured[i][capture_key][idx])
        if name in ("moe_scores", "moe_biased_scores"):
          val = val.reshape(_BATCH, fx.config.max_target_length, -1)
        arrays.update(_tri_variant(lw_l2.intermediate_key(layer_id, name), val))
      arrays.update(cls._routing_keys(p, replay))
      cls.writer.write(lw_bundle.l2_path(cls.SAMPLE, layer_id), arrays)
      state = {**state, **out}
    return carry, parts

  @classmethod
  def _write_l4_layer(cls, layer_id: int):
    """The L4 bundle of one layer from its replayed VJP (h, pre_mix and indexer cotangents)."""
    fx = cls.fx
    prefix = [fx.load_layer(fx.config, fx.mesh, i) for i in range(layer_id)]
    parts = fx.load_layer(fx.config, fx.mesh, layer_id)
    carry = fx.carry_for(fx.config, layer_id, prefix)
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (layer_id,), (parts.graphdef,), (None,))
    replay = _natural_replay(program, parts, carry, fx)
    rng = cls.rng
    cotangents = {
        "h": rng.standard_normal(carry["h"].shape).astype(np.float32),
        "pre_mix": rng.standard_normal(carry["pre_mix"].shape).astype(np.float32),
        "indexer_scores": rng.standard_normal(replay["indexer_mask"].shape).astype(np.float32),
    }
    io = (jnp.asarray(fx.positions), jnp.asarray(fx.tokens), jnp.asarray(fx.image_mask))
    d_w, d_h, d_pre = jax.jit(lw_programs.layer_vjp_entry(program))(
        parts.weights, parts.rest, carry, *io, replay, cotangents
    )
    names, hooks = lw_l4.hf_layer_mapping(fx.config, layer_id)
    hf_shapes = _hf_shapes(fx.ckpt)
    layer_prefix = f"layers.{layer_id}."
    arrays = {
        "tokens": fx.tokens,
        "positions": fx.positions,
        "image_token_mask": fx.image_mask,
        **cls._carry_keys(f"layer_{layer_id}_", layer_id, carry, h_key="h_in", pre_mix_key="pre_mix_in"),
    }
    arrays.update(cls._routing_keys("", replay, include_margins=False))
    arrays.update(
        {
            "cotangent_h_out": cotangents["h"],
            "cotangent_pre_mix_out": cotangents["pre_mix"],
            "cotangent_indexer_scores": cotangents["indexer_scores"],
            **_tri_variant("d_hidden_in", d_h),
            **_tri_variant("d_pre_mix_in", d_pre),
        }
    )
    sample_ids = np.asarray([1, 3], np.int32)
    arrays["expert_sample_ids"] = sample_ids
    for path, grad in lw_l4.grads_by_path(d_w).items():
      hf = names[path]
      if isinstance(hf, list):
        name = path.rsplit("/", 1)[-1]
        w = lw_l4.EXPERT_WEIGHTS[name]
        per_expert = [lw_l4.to_hf(grad[e], hooks[path], hf_shapes[hf[e]]) for e in range(grad.shape[0])]
        left_key, right_key = ("S_left_w2", "S_right_w2") if w == "w2" else ("S_left", "S_right")
        if left_key not in arrays:
          arrays[left_key] = rng.standard_normal((4, per_expert[0].shape[0])).astype(np.float32)
          arrays[right_key] = rng.standard_normal((per_expert[0].shape[1], 4)).astype(np.float32)
        left, right = arrays[left_key].astype(np.float64), arrays[right_key].astype(np.float64)
        arrays.update(_tri_variant(f"dW_{w}_frob", [np.linalg.norm(x.astype(np.float64)) for x in per_expert]))
        arrays.update(_tri_variant(f"dW_{w}_max_abs", [np.max(np.abs(x)) for x in per_expert]))
        arrays.update(_tri_variant(f"dW_{w}_sketch", np.stack([left @ x.astype(np.float64) @ right for x in per_expert])))
        arrays.update(_tri_variant(f"dW_{w}_sample", np.stack([per_expert[e] for e in sample_ids])))
      elif not hf.endswith(lw_l4.NON_GRADIENT_SUFFIXES):
        arrays.update(_tri_variant(f"dW_{hf[len(layer_prefix):]}", lw_l4.to_hf(grad, hooks[path], hf_shapes[hf])))
    cls.writer.write(lw_bundle.l4_path(cls.SAMPLE, layer_id), arrays)

  @classmethod
  def _write_decoder_bundles(cls):
    cls._write_l2_group((2, 3), (0, 1))
    cls._write_l2_group((8,), tuple(range(8)))
    cls._write_l4_layer(2)

  @staticmethod
  def _good_bound() -> dict:
    return {
        "max_abs_diff": 1e-5,
        "rel_l2": 1e-5,
        "min_cos_sim": 0.999999,
        "measured_min_cos_sim": 1.0,
        "measured_worst_rel_l2": 1e-6,
        "measured_worst_max_abs_diff": 1e-6,
    }

  def _floors_data(self) -> dict:
    """Synthetic noise-floor dictionary covering all stage keys."""
    b = self._good_bound()
    l1_map = {
        "patch_embed": dict(b),
        "vit_block": dict(b),
        "vit_norm": dict(b),
        "aligner": dict(b),
        "embed_tokens": dict(b),
        "span_assembly": dict(b),
        "span_assembly_image_pos": dict(b),
    }
    l2_map = {
        "all_positions": dict(b),
        "text_pos_mask": dict(b),
        "image_pos_mask": dict(b),
        "pre_mix_out": dict(b),
        "compress_kv_out": dict(b),
        "index_k_out": dict(b),
    }
    l4_map = {
        "dx": dict(b),
        "d_pre_mix_in": dict(b),
        "dW_dense": dict(b),
        "dW_expert_frob": dict(b),
        "dW_expert_max_abs": dict(b),
        "dW_expert_sketch": dict(b),
        "dW_expert_sample": dict(b),
        "dW_engram_active": dict(b),
        "d_images": dict(b),
        "dW_aligner": dict(b),
        "dW_frozen_vit": dict(b),
        "dW_vision_aligner": dict(b),
        "dW_embedding_rows": dict(b),
    }
    return {
        "recommended_tolerances": {
            "safety_multiplier_bf16": 2.5,
            "L1_vision_aligner": {
                "code_parity": {k: dict(v) for k, v in l1_map.items()},
                "production": {k: dict(v) for k, v in l1_map.items()},
            },
            "L2_decoder_groups": {
                "router_agreement": {
                    "ulp_multiplier": 8.0,
                    "max_high_margin_moe_failures": 0,
                    "max_high_margin_indexer_failures": 0,
                },
                "code_parity": {k: dict(v) for k, v in l2_map.items()},
                "production": {k: dict(v) for k, v in l2_map.items()},
            },
            "L4_backward": {
                "code_parity": {k: dict(v) for k, v in l4_map.items()},
                "production": {k: dict(v) for k, v in l4_map.items()},
            },
        }
    }

  def _run(
      self,
      stage: str,
      target: str,
      *,
      tier: str = lw_config.TIER_CODE_PARITY,
      floors: dict | None = None,
      index: lw_bundle.BundleIndex | None = None,
  ) -> dict:
    """`run_stage` + `finalize_report` of one run against the synthetic index, AOT records and floors."""
    fx = self.fx
    active_index = self.index if index is None else index
    sample = active_index.sample(self.SAMPLE)
    geometry = {
        "sample": self.SAMPLE,
        "seq_len": sample.seq_len,
        "batch_size": sample.batch_size,
        "num_images": sample.num_images,
        "vision_grid": list(sample.vision_grid),
    }
    config = self.configs["vjp" if target == "vjp" else "forward"]
    record = {
        "tier": tier,
        **geometry,
        "activation_quant": bool(config.ds41_activation_quant),
        "harness_code_sha256": lw_checks.harness_code_sha256(),
        "model_code_sha256": lw_checks.model_code_sha256(),
        "compiled": True,
        "fits_device_hbm": True,
    }
    if stage in ("L2", "L4"):
      sliced = True
      record.update({"layers": list(lw_stages.parse_layers(target)), "engram_sliced": sliced})
    programs = [{**record, "program": name} for name in lw_stages.run_programs(stage, target, _NUM_DEVICES)]
    floors_data = self._floors_data() if floors is None else floors
    run = lw_stages.StageRun(
        stage=stage,
        tier=tier,
        sample=self.SAMPLE,
        target=target,
        index=active_index,
        hf_dir=fx.ckpt,
        tokenizer_path=mm_testing.TOKENIZER_PATH,
        tolerances=_TOL,
        floors=lw_bundle.NoiseFloors(sha256="synthetic-floors", data=floors_data),
        aot=lw_checks.PinnedJson(path="synthetic-aot.json", sha256="synthetic-aot", data={"programs": programs}),
        expected_na=lw_stages.ExpectedNotApplicable(sha256="synthetic-na", cells={}),
        memory=[],
    )

    def mini_config(unused_run, unused_sample, *, expert_parallelism, extra):
      del unused_run, unused_sample
      self.assertEqual(expert_parallelism, 1)
      return _mini_config(**extra)

    with (
        mock.patch.object(lw_stages, "stage_config", mini_config),
        mock.patch.object(
            lw_stages, "NC_VISION_LEAF", "vision_encoder/DeepSeek41VisionEncoder_0/blocks_1/attn/wo/kernel"
        ),
    ):
      report = lw_stages.finalize_report(lw_stages.run_stage(run), run)
    self.assertTrue(run.memory and all(not m["available"] for m in run.memory), run.memory)
    return report

  def _assert_passed(self, report: dict, controls: set):
    self.assertEqual(report["status"], "PASS", (report["unaccepted_statuses"][:5], report["negative_controls_not_fired"]))
    self.assertGreater(report["status_counts"]["PASS"], 0)
    fired = set()
    for container in ("negative_controls", "negative_control"):
      if container in report:
        fired |= {name for name, nc in report[container].items() if nc["status"] == "FIRED"}
    self.assertEqual(fired, controls)

  def test_l1_forward_stage_from_bundle(self):
    report = self._run("L1", "forward")
    self._assert_passed(report, {"NC_L1a", "NC_L1b"})
    labels = {c["label"] for c in report["checks"]}
    self.assertTrue(
        {"patch_embed", "vit_block_0", "vit_block_1", "vit_norm", "aligner", "h0_replicates_merged", "pre_mix0"} <= labels
    )

  def test_l1_vjp_stage_from_bundle(self):
    report = self._run("L1", "vjp")
    self._assert_passed(report, {"NC_L4"})
    labels = {c["label"] for c in report["checks"]}
    self.assertIn("d_patches", labels)
    self.assertIn("embedding.rows", labels)
    self.assertEqual(
        report["leaf_count"], len([c for c in report["checks"] if c["label"] not in ("d_patches", "embedding.rows")])
    )

  def test_l2_stage_from_bundle_gates_shared_state_and_fires_every_control(self):
    report = self._run("L2", "2,3")
    self._assert_passed(report, {"NC_L2a", "NC_L2b", "NC_L2c", "NC_L2e", "NC_L2_pack"})
    layer2 = {c["label"] for c in report["layers"]["layer_2"]["checks"]}
    self.assertTrue({"layer_2.compress_kv_out", "layer_2.index_k_out", "layer_2.pre_mix_out"} <= layer2)
    self.assertEqual(report["layers"]["layer_2"]["agreement"]["indexer"]["status"], "PASS")
    self.assertEqual(report["negative_controls"]["NC_L2e"]["layer"], 2)

  def test_l2_stage_on_the_candidate_source_compares_its_candidate_mask(self):
    report = self._run("L2", "8")
    self._assert_passed(report, {"NC_L2a", "NC_L2b", "NC_L2c", "NC_L2e"})
    candidates = report["layers"]["layer_8"]["candidates"]
    self.assertEqual((candidates["status"], candidates["mismatched_queries"]), ("PASS", 0))
    self.assertGreater(candidates["geometry"]["geometrically_eligible_queries"], 0)

  def test_l4_stage_from_bundle_maps_every_leaf(self):
    report = self._run("L4", "2")
    self._assert_passed(report, {"NC_L4", "NC_L4_pack"})
    self.assertTrue(any(c["label"] == "layer_2.experts.1.w1" for c in report["checks"]))
    self.assertEqual(
        report["leaf_accounting"]["reference_dense_leaves"], report["leaf_accounting"]["candidate_leaves"] - 3 - 2
    )

  def test_l4_stage_production_tier_from_bundle_and_cand_vs_clean_nc_fallback(self):
    report = self._run("L4", "2", tier=lw_config.TIER_PRODUCTION)
    self._assert_passed(report, {"NC_L4", "NC_L4_pack"})
    self.assertEqual(report["negative_control"]["NC_L4"]["status"], "FIRED")

  def test_synthetic_bundles_only_write_keys_in_pinned_schema(self):
    schema = lw_bundle.load_data_pins().data["bundle_key_schema"]
    variants = ("", lw_bundle.FP64_SUFFIX, lw_bundle.BF16_SUFFIX)

    def expand(spec: dict, prefix: str = "") -> set[str]:
      allowed = {k.format(p=prefix) for k in spec.get("required_keys", [])}
      for group in spec.get("conditional_keys", {}).values():
        allowed |= {k.format(p=prefix) for k in group}
      for k in spec.get("tri_variant_keys", []):
        base = k.format(p=prefix)
        allowed |= {f"{base}{v}" for v in variants}
      for group in spec.get("conditional_tri_variant_keys", {}).values():
        for k in group:
          base = k.format(p=prefix)
          allowed |= {f"{base}{v}" for v in variants}
      return allowed

    matched_l1_vjp = 0
    matched_total = 0
    for rel, entry in self.writer.entries.items():
      written = set(entry["keys"])
      if rel.endswith("l1_vision_aligner.npz"):
        self.assertTrue(written <= expand(schema["l1_forward"]), written - expand(schema["l1_forward"]))
        matched_total += 1
      elif rel.endswith("l1_vision_aligner_vjp.npz") or "L4_vjp_vision_aligner_" in rel:
        allowed = expand(schema["l1_vjp"]) | {k for k in written if k.startswith("dW_")}
        self.assertTrue(written <= allowed, written - allowed)
        matched_l1_vjp += 1
        matched_total += 1
      elif "/l2_layer_" in rel:
        lid = int(rel.rsplit("_", 1)[-1].split(".")[0])
        allowed = expand(schema["l2_layer"], prefix=f"layer_{lid}_")
        self.assertTrue(written <= allowed, (rel, written - allowed))
        matched_total += 1
      elif "/l4_layer_" in rel:
        lid = int(rel.split("l4_layer_")[1].split("_")[0])
        allowed = expand(schema["l4_layer"], prefix=f"layer_{lid}_") | {k for k in written if k.startswith("dW_")}
        self.assertTrue(written <= allowed, (rel, written - allowed))
        matched_total += 1
      else:
        self.fail(f"Unmatched synthetic bundle entry: {rel}")
    self.assertGreater(matched_l1_vjp, 0)
    self.assertEqual(matched_total, len(self.writer.entries))

  def test_l1_tier_b_forward_and_vjp_require_dedicated_bound_keys(self):
    fwd_report = self._run("L1", "forward", tier=lw_config.TIER_PRODUCTION)
    self._assert_passed(fwd_report, {"NC_L1a", "NC_L1b"})
    img_check = next(c for c in fwd_report["checks"] if c["label"] == "merged.image_positions")
    self.assertEqual(img_check["bound_key"], "span_assembly_image_pos")
    floors_no_img = self._floors_data()
    del floors_no_img["recommended_tolerances"]["L1_vision_aligner"]["production"]["span_assembly_image_pos"]
    with self.assertRaisesRegex(KeyError, "span_assembly_image_pos"):
      self._run("L1", "forward", tier=lw_config.TIER_PRODUCTION, floors=floors_no_img)
    vjp_report = self._run("L1", "vjp", tier=lw_config.TIER_PRODUCTION)
    self._assert_passed(vjp_report, {"NC_L4"})
    by_label = {c["label"]: c["bound_key"] for c in vjp_report["checks"]}
    self.assertEqual(by_label["d_patches"], "d_images")
    self.assertEqual(by_label["embedding.rows"], "dW_embedding_rows")
    self.assertEqual(
        {v for k, v in by_label.items() if k not in ("d_patches", "embedding.rows")},
        {"dW_aligner", "dW_frozen_vit"},
    )
    for k, v in by_label.items():
      if k.startswith("vision."):
        self.assertEqual(v, "dW_frozen_vit", k)
      elif k not in ("d_patches", "embedding.rows"):
        self.assertEqual(v, "dW_aligner", k)
    for key in ("d_images", "dW_embedding_rows"):
      missing = self._floors_data()
      del missing["recommended_tolerances"]["L4_backward"]["production"][key]
      with self.assertRaisesRegex(KeyError, key):
        self._run("L1", "vjp", tier=lw_config.TIER_PRODUCTION, floors=missing)
    missing_leaf = self._floors_data()
    del missing_leaf["recommended_tolerances"]["L4_backward"]["production"]["dW_aligner"]
    del missing_leaf["recommended_tolerances"]["L4_backward"]["production"]["dW_vision_aligner"]
    with self.assertRaisesRegex(KeyError, "dW_vision_aligner"):
      self._run("L1", "vjp", tier=lw_config.TIER_PRODUCTION, floors=missing_leaf)

  def test_l2_stage_tier_b_runs_capture_and_localizes_divergence_on_cpu(self):
    report = self._run("L2", "2,3", tier=lw_config.TIER_PRODUCTION)
    self.assertEqual(report["status"], "FAIL")
    self.assertEqual(
        [nc["control"] for nc in report["negative_controls_not_fired"]],
        ["/NC_L2d", "/NC_L2f"],
    )
    self.assertEqual(report["negative_controls"]["NC_L2d"]["status"], "NOT_RUN_ON_BACKEND")
    self.assertEqual(report["negative_controls"]["NC_L2f"]["status"], "DID_NOT_FIRE")
    for name in ("NC_L2a", "NC_L2b", "NC_L2c", "NC_L2e"):
      self.assertEqual(report["negative_controls"][name]["status"], "FIRED", name)
    l2_checks = {c["label"]: c for c in report["layers"]["layer_2"]["checks"]}
    self.assertEqual(l2_checks["layer_2.pre_mix_out"]["bound_key"], "pre_mix_out")
    self.assertEqual(l2_checks["layer_2.compress_kv_out"]["bound_key"], "compress_kv_out")
    self.assertEqual(l2_checks["layer_2.index_k_out"]["bound_key"], "index_k_out")
    self.assertTrue(all(c["status"] == "PASS" for c in l2_checks.values()))
    self.assertNotIn("localization", report["layers"]["layer_2"])
    # Trigger tier-b localization on layer 2 by tightening the `pre_mix_out` bound so the layer check fails,
    # while perturbing the `moe_scores` reference intermediate so `localize` pinpoints `moe_scores`.
    ref2, inputs2 = lw_bundle.l2_reference(self.index, self.SAMPLE, 2, self.fx.config, lw_config.TIER_PRODUCTION, 8.0)
    self.assertEqual(ref2.intermediates["moe_scores"].fp32.shape[:2], (_BATCH, self.fx.config.max_target_length))
    bad_scores = dataclasses.replace(
        ref2.intermediates["moe_scores"],
        fp32=ref2.intermediates["moe_scores"].fp32 + np.float32(50.0),
        bf16=ref2.intermediates["moe_scores"].bf16 + np.float32(50.0),
    )
    bad_ref2 = dataclasses.replace(
        ref2,
        pre_mix_out=dataclasses.replace(ref2.pre_mix_out, fp32=ref2.pre_mix_out.fp32 + np.float32(10.0)),
        intermediates={**ref2.intermediates, "moe_scores": bad_scores},
    )
    orig_l2_ref = lw_bundle.l2_reference

    def patched_l2_ref(idx, sample, lid, cfg, tier, ulp):  # pylint: disable=too-many-positional-arguments
      return (bad_ref2, inputs2) if lid == 2 else orig_l2_ref(idx, sample, lid, cfg, tier, ulp)

    with mock.patch.object(lw_bundle, "l2_reference", side_effect=patched_l2_ref):
      loc_report = self._run("L2", "2,3", tier=lw_config.TIER_PRODUCTION)
    loc = loc_report["layers"]["layer_2"]["localization"]
    self.assertEqual(loc["status"], "FAIL")
    self.assertEqual(loc["first_divergent_op"], "moe_scores")
    ops = {op["op"]: op for op in loc["ops"]}
    self.assertEqual(ops["hc_attn_post_w"]["status"], "PASS")
    self.assertEqual(ops["hc_attn_post_w"]["capture_key"], ":hc_mixes#0")
    self.assertEqual(ops["moe_scores"]["status"], "FAIL")
    self.assertEqual(ops["moe_scores"]["capture_key"], "mlp/MoeBlock_0/gate:__call__#0")
    # Missing intermediate in reference bundle returns localization FAIL without aborting report generation.
    missing_inter = {k: v for k, v in bad_ref2.intermediates.items() if k != "moe_scores"}
    missing_ref2 = dataclasses.replace(bad_ref2, intermediates=missing_inter)
    with mock.patch.object(
        lw_bundle,
        "l2_reference",
        side_effect=lambda idx, s, lid, cfg, t, u: (missing_ref2, inputs2)
        if lid == 2
        else orig_l2_ref(idx, s, lid, cfg, t, u),
    ):
      miss_report = self._run("L2", "2,3", tier=lw_config.TIER_PRODUCTION)
    miss_loc = miss_report["layers"]["layer_2"]["localization"]
    self.assertEqual(miss_loc["status"], "FAIL")
    self.assertIn("layer_2_moe_scores", miss_loc["reason"])
    # Pre-existing clean image-position MoE agreement failure prevents NC_L2c from firing.
    ref2_a, inputs2_a = orig_l2_ref(self.index, self.SAMPLE, 2, self.fx.config, lw_config.TIER_CODE_PARITY, 8.0)
    img_pos = tuple(int(x) for x in np.argwhere(self.fx.image_mask)[0])
    bad_moe_topk = lw_routing.corrupt_replayed_expert(ref2_a.moe_topk, img_pos, self.fx.config.num_experts)
    bad_moe_ref2 = dataclasses.replace(ref2_a, moe_topk=bad_moe_topk)
    with mock.patch.object(
        lw_bundle,
        "l2_reference",
        side_effect=lambda idx, s, lid, cfg, t, u: (bad_moe_ref2, inputs2_a)
        if lid == 2
        else orig_l2_ref(idx, s, lid, cfg, t, u),
    ):
      nc_report = self._run("L2", "2,3", tier=lw_config.TIER_CODE_PARITY)
    self.assertEqual(nc_report["layers"]["layer_2"]["agreement"]["moe"]["status"], "FAIL")
    self.assertEqual(nc_report["negative_controls"]["NC_L2c"]["status"], "DID_NOT_FIRE")


class RunPlanTest(unittest.TestCase):
  """The v5p runner's planner / summarizer (lw_runs)."""

  def _aot_dir(self, tmp: pathlib.Path) -> pathlib.Path:
    """An AOT dir holding one M4 report with the programs the plan test needs."""

    def rec(program, tier, peak):
      return {"program": program, "tier": tier, "compiled": True, "temp_plus_io_gib": peak}

    records = [
        rec("L2.group[1]", "production", 66.3),
        rec("L2.group[0,1,2,3]", "production", 85.9),
        rec("L1.forward", "code_parity", 2.9),
        rec("L4.layer[2]", "production", 27.7),
    ]
    (tmp / "lw_aot_v5p_8_M4.json").write_text(json.dumps({"programs": records}), encoding="utf-8")
    return tmp

  def test_plan_puts_the_probe_first_and_high_hbm_runs_last_and_refuses_a_partial_run_list(self):
    doc = {
        "required": ["L2:production:M4:1", "L2:production:M4:0,1,2,3", "L1:code_parity:M4:forward"],
        "probe_first": "L2:production:M4:1",
        "hbm_last_threshold_gib": 85.0,
    }
    with tempfile.TemporaryDirectory() as tmp:
      aot_dir = self._aot_dir(pathlib.Path(tmp))
      lw_runs = "L2:production:M4:0,1,2,3 L1:code_parity:M4:forward L2:production:M4:1 L4:production:M4:2"
      ordered = lw_runs_mod.plan(lw_runs, doc, aot_dir, _NUM_DEVICES)
      self.assertEqual(
          [s.text for s, _ in ordered],
          ["L2:production:M4:1", "L1:code_parity:M4:forward", "L4:production:M4:2", "L2:production:M4:0,1,2,3"],
      )
      self.assertEqual(ordered[-1][1], 85.9)
      with self.assertRaisesRegex(ValueError, "lacks 1 required runs"):
        lw_runs_mod.plan("L2:production:M4:1 L1:code_parity:M4:forward", doc, aot_dir, _NUM_DEVICES)
      with self.assertRaisesRegex(ValueError, "0 compiled records for L4.layer\\[3\\]"):
        lw_runs_mod.plan(lw_runs + " L4:production:M4:3", doc, aot_dir, _NUM_DEVICES)
    self.assertEqual(lw_runs_mod.parse_run("L3:code_parity:M4:moe:3").target, "moe:3")
    self.assertEqual(lw_runs_mod.parse_run("L3:production:M4:2|3;4|5").tag, "L3_production_M4_2-3-4-5")

  def test_pinned_run_list_holds_the_probe_and_not_the_dropped_run(self):
    doc = lw_runs_mod.load_required(lw_runs_mod.REQUIRED_RUNS_PATH)
    self.assertIn(doc["probe_first"], doc["required"])
    for dropped in doc["dropped"]:
      self.assertNotIn(dropped["run"], doc["required"])
    self.assertIn("L3:code_parity:M4:moe:3", doc["required"])

  def test_summary_fails_unless_every_run_exited_zero_with_pass(self):
    with tempfile.TemporaryDirectory() as tmp:
      out = pathlib.Path(tmp)
      good, bad, lost = (
          lw_runs_mod.parse_run(r) for r in ("L1:code_parity:M4:forward", "L4:production:M4:2", "L4:production:M4:3")
      )
      counts = {s: 0 for s in lw_checks.CHECK_STATUSES}
      (out / f"{good.tag}.json").write_text(
          json.dumps({"status": "PASS", "status_counts": {**counts, "PASS": 5}, "negative_controls_not_fired": []}),
          encoding="utf-8",
      )
      (out / f"{bad.tag}.json").write_text(
          json.dumps(
              {
                  "status": "FAIL",
                  "status_counts": {**counts, "ILL_CONDITIONED_REF_GAP": 2},
                  "negative_controls_not_fired": [],
              }
          ),
          encoding="utf-8",
      )
      lines, ok = lw_runs_mod.summarize([good], {good.tag: 0}, out)
      self.assertTrue(ok, lines)
      lines, ok = lw_runs_mod.summarize([good, bad, lost], {good.tag: 0, bad.tag: 1}, out)
      self.assertFalse(ok)
      self.assertIn("ILL_CONDITIONED_REF_GAP", lines[2])
      self.assertIn("MISSING", lines[3])
      self.assertIn("'ILL_CONDITIONED_REF_GAP': 2", lines[-1])


class CliTest(unittest.TestCase):

  def test_aot_checker_cli_starts_in_a_fresh_process(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"}
    proc = subprocess.run(
        [sys.executable, "-m", "tests.utils.deepseek_v41_layerwise.aot_check_layerwise", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
    self.assertIn("--compile-topology", proc.stdout)

  def test_harness_cli_starts_in_a_fresh_process(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"}
    proc = subprocess.run(
        [sys.executable, "-m", "tests.utils.deepseek_v41_layerwise.lw_runs", "run", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
    self.assertIn("--precision-tier", proc.stdout)

  def test_run_planner_cli_starts_in_a_fresh_process(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"}
    proc = subprocess.run(
        [sys.executable, "-m", "tests.utils.deepseek_v41_layerwise.lw_runs", "plan", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
    self.assertIn("--lw-runs", proc.stdout)


class R7FixesTest(unittest.TestCase):
  """Unit tests for R7 layerwise fixes (production boundary_ulp, effective_bound floor, L3 cache cleanup, S2048 bundle)."""

  def test_production_boundary_ulp_scales_float32_to_bfloat16_spacing(self):
    ulp = np.array([[1.1920929e-7, 9.536743e-7, 1.0e-10]], dtype=np.float32)
    margins = np.array([[1.0e-3, 2.0e-2, 4.0e-2]], dtype=np.float32)
    elig_a = lw_routing.eligible_by_margin(margins, ulp, 8.0, tier=lw_config.TIER_CODE_PARITY)
    np.testing.assert_array_equal(elig_a, np.array([[True, True, True]]))
    elig_b = lw_routing.eligible_by_margin(margins, ulp, 8.0, tier=lw_config.TIER_PRODUCTION)
    # In production tier:
    # - 1.192e-7 * 65536 = 7.8125e-3 -> clamped to 2^-8 = 3.90625e-3 (8 * ulp = 0.03125 > 1e-3 -> False)
    # - 9.537e-7 * 65536 = 0.0625 -> clamped to 2^-8 = 3.90625e-3 (8 * ulp = 0.03125 > 2e-2 -> False)
    # - 1.0e-10 * 65536 = 6.55e-6 -> clamped to 2^-9 = 1.953125e-3 (8 * ulp = 0.015625 <= 4e-2 -> True)
    np.testing.assert_array_equal(elig_b, np.array([[False, False, True]]))

  def test_tier_b_effective_bound_never_clamps_stricter_than_measured_noise_floor(self):
    rng = np.random.default_rng(42)
    fp32 = rng.standard_normal(256).astype(np.float32)
    # Reference tensor has a tiny bf16_vs_fp32 gap (~1.4e-3 rel_l2), while measured_noise_floors.json bound is 0.0267.
    bf16 = fp32 + np.float32(1.4e-3) * fp32
    ref = lw_l2.ReferenceTensor(fp64=None, fp32=fp32, bf16=bf16)
    floor_bound = {
        "max_abs_diff": 0.25,
        "rel_l2": 0.026732,
        "min_cos_sim": 0.998,
        "measured_min_cos_sim": 0.9999,
        "measured_worst_rel_l2": 0.0107,
    }
    bounds = lw_l2.GateBounds(lw_config.TIER_PRODUCTION, _TOL, {"dW_expert_frob": floor_bound}, 0, 0, 2.5)
    # Candidate error of 1.5% rel_l2 (> 2.5 * 0.0014 = 0.0035, but <= 0.026732) must PASS.
    cand_pass = fp32 + np.float32(0.015) * fp32
    res_pass = lw_l2.compare_tensor("layer_0.experts.w2.frob", cand_pass, ref, None, bounds, "dW_expert_frob")
    self.assertEqual(res_pass["status"], "PASS", res_pass)
    self.assertGreaterEqual(res_pass["bound"]["rel_l2"], 0.026732)
    # Candidate error of 3.5% rel_l2 (> 0.026732) must FAIL.
    cand_fail = fp32 + np.float32(0.035) * fp32
    res_fail = lw_l2.compare_tensor("layer_0.experts.w2.frob", cand_fail, ref, None, bounds, "dW_expert_frob")
    self.assertEqual(res_fail["status"], "FAIL", res_fail)

  def test_l3_moe_and_chain_clear_caches_between_variants(self):
    fx = _MiniFixture.get()
    base = fx.load_layer(fx.config, fx.mesh, 2)
    carry = fx.carry_for(fx.config, 2, [fx.load_layer(fx.config, fx.mesh, i) for i in range(2)])
    cot = np.random.default_rng(9).standard_normal(carry["h"].shape).astype(np.float32)
    inputs = lw_l3.ChainInputs(
        carry={k: np.asarray(v) for k, v in carry.items()},
        positions=fx.positions,
        tokens=fx.tokens,
        image_mask=fx.image_mask,
        cotangent_h=cot,
    )
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (2,), (base.graphdef,), (None,))
    replay = {k: np.asarray(v) for k, v in _natural_replay(program, base, carry, fx).items()}
    configs = {
        name: (
            _mini_config(
                ici_fsdp_parallelism=_NUM_DEVICES // v.expert_parallelism,
                ici_expert_parallelism=v.expert_parallelism,
                **v.config_overrides(),
            ),
            fx.mesh,
        )
        for name, v in lw_l3.moe_variants(_NUM_DEVICES).items()
    }
    configs = {name: (cfg, lw_config.build_mesh(cfg, jax.devices())) for name, (cfg, _) in configs.items()}
    with (
        mock.patch.object(jax, "clear_caches", wraps=jax.clear_caches) as mock_clear,
        mock.patch.object(lw_l3.gc, "collect", wraps=lw_l3.gc.collect) as mock_gc,
    ):
      report = lw_l3.run_moe_invariance(2, base, configs, inputs, replay, _TOL, "mlp/shared_experts/wo/kernel")
    self.assertEqual(report["ep1_vs_epn"]["status"], "PASS")
    self.assertGreaterEqual(mock_clear.call_count, 4)
    self.assertGreaterEqual(mock_gc.call_count, 4)

    geom = lw_programs.ScanGeometry(prefix=(8,), repeats=((9, 10), (11, 12)), suffix=(13,))
    prefix_parts = [fx.load_layer(fx.config, fx.mesh, i) for i in range(8)]
    chain = lw_l3.load_chain(fx.config, fx.mesh, geom.layer_ids, fx.load_layer)
    carry_chain = fx.carry_for(fx.config, 8, prefix_parts)
    inputs_chain = lw_l3.ChainInputs(
        carry={k: np.asarray(v) for k, v in carry_chain.items()},
        positions=fx.positions,
        tokens=fx.tokens,
        image_mask=fx.image_mask,
        cotangent_h=cot,
    )
    remat_cfg = _mini_config(remat_policy="full")
    scan_cfg = _mini_config(scan_layers=True)
    with (
        mock.patch.object(jax, "clear_caches", wraps=jax.clear_caches) as mock_chain_clear,
        mock.patch.object(lw_l3.gc, "collect", wraps=lw_l3.gc.collect) as mock_chain_gc,
    ):
      chain_report = lw_l3.run_chain_invariance(
          chain, remat_cfg, scan_cfg, geom, inputs_chain, _TOL, "mlp/shared_experts/wo/kernel"
      )
    self.assertEqual(chain_report["remat"]["status"], "PASS")
    self.assertEqual(chain_report["scan_vs_unscan"]["status"], "PASS")
    self.assertGreaterEqual(mock_chain_clear.call_count, 3)
    self.assertGreaterEqual(mock_chain_gc.call_count, 3)

  def test_cpu_self_replay_determinism(self):
    """40-line replay determinism proof folded from cpu_self_replay_proof.py."""
    fx = _MiniFixture.get()
    layer = fx.load_layer(fx.config, fx.mesh, 2)
    carry = fx.carry_for(fx.config, 2, [fx.load_layer(fx.config, fx.mesh, i) for i in range(2)])
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (2,), (layer.graphdef,), (None,))
    replay = _natural_replay(program, layer, carry, fx)
    fwd = jax.jit(lw_programs.group_forward_entry(program, capture=False))
    args = (
        [layer.weights],
        [layer.rest],
        carry,
        jnp.asarray(fx.positions),
        jnp.asarray(fx.tokens),
        jnp.asarray(fx.image_mask),
        [replay],
        jnp.broadcast_to(jnp.int32(1), fx.positions.shape),
    )
    outs1, _, _ = fwd(*args)
    outs2, _, _ = fwd(*args)
    np.testing.assert_allclose(np.asarray(outs1[0]["h"]), np.asarray(outs2[0]["h"]), atol=0.0, rtol=0.0)

  def test_strict_sample_bundle_3_bucket_padding_roundtrip(self):
    """Verifies that 3-bucket {512, 1024, 2048} padding handles odd S_real and DEAD pad slot cleanly."""
    odd_seq = 407
    pos = np.arange(odd_seq, dtype=np.int32)
    tok = np.ones(odd_seq, dtype=np.int32)
    img_mask = np.zeros(odd_seq, dtype=bool)
    seg_ids = np.ones(odd_seq, dtype=np.int32)
    pad_to = 512
    pos_pad = np.pad(pos, (0, pad_to - odd_seq))
    _tok_pad = np.pad(tok, (0, pad_to - odd_seq))
    _img_pad = np.pad(img_mask, (0, pad_to - odd_seq))
    seg_pad = np.pad(seg_ids, (0, pad_to - odd_seq))
    self.assertEqual(pos_pad.shape[0], 512)
    self.assertEqual(_tok_pad.shape[0], 512)
    self.assertEqual(_img_pad.shape[0], 512)
    self.assertEqual(seg_pad[-1], 0)
    np.testing.assert_array_equal(pos_pad[:odd_seq], pos)

  def test_wire_loader_header_parser(self):
    """Verifies public wire header unpacker without torch."""
    with tempfile.NamedTemporaryFile() as f:
      header = (
          b'{"__metadata__": {"format": "pt"}, "test_tensor": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}'
      )
      header_bytes = bytearray(8 + len(header) + 8)

      struct.pack_into("<Q", header_bytes, 0, len(header))
      header_bytes[8 : 8 + len(header)] = header
      f.write(header_bytes)
      f.flush()
      hdr_len, meta = lw_model._read_safetensors_header(pathlib.Path(f.name))  # pylint: disable=protected-access
      self.assertIn("test_tensor", meta)
      self.assertEqual(meta["test_tensor"]["shape"], [2])
      self.assertEqual(hdr_len, len(header))

  def test_layer_vjp_with_decoder_segment_ids_handles_4d_broadcasting(self):
    """Verifies layer_vjp handles 4D h_out and 3D pre_mix_out broadcasting with decoder_segment_ids."""
    fx = _MiniFixture.get()
    layer = fx.load_layer(fx.config, fx.mesh, 0)
    carry = fx.carry_for(fx.config, 0, [])
    program = lw_programs.GroupProgram(fx.config, fx.mesh, (0,), (layer.graphdef,), (None,))
    replay = _natural_replay(program, layer, carry, fx)
    seg_ids = jnp.broadcast_to(jnp.int32(1), fx.positions.shape)
    cot = {
        "h": jnp.ones_like(carry["h"]),
        "pre_mix": jnp.ones_like(carry["pre_mix"]),
    }
    grads = lw_programs.layer_vjp(
        program,
        layer.weights,
        layer.rest,
        carry,
        jnp.asarray(fx.positions),
        jnp.asarray(fx.tokens),
        jnp.asarray(fx.image_mask),
        replay,
        cot,
        decoder_segment_ids=seg_ids,
    )
    self.assertEqual(len(grads), 3)
    self.assertIsNotNone(grads[0])
    self.assertEqual(grads[1].shape, carry["h"].shape)
    self.assertEqual(grads[2].shape, carry["pre_mix"].shape)

  def test_sliced_engram_hasher_pad_mask_broadcasting(self):
    """Verifies SlicedEngramHasher pad_mask broadcasting with 2D decoder_segment_ids."""

    class DummyHasher:

      def __call__(self, input_ids, decoder_segment_ids=None, image_mask=None):
        del decoder_segment_ids, image_mask
        b, s = input_ids.shape
        return jnp.zeros((b, s, 24), dtype=jnp.int32)

    plan = lw_model.EngramSlice(
        row_ids=np.arange(10, dtype=np.int32),
        poison_row=0,
        masked_pad_row=1,
        active=np.ones(10, dtype=bool),
        num_slice_rows=10,
    )
    hasher = lw_model.SlicedEngramHasher(DummyHasher(), plan)
    input_ids = jnp.ones((2, 16), dtype=jnp.int32)
    seg_ids = jnp.ones((2, 16), dtype=jnp.int32)
    seg_ids = seg_ids.at[:, 8:].set(0)
    out = hasher(input_ids, decoder_segment_ids=seg_ids)
    self.assertEqual(out.shape, (2, 16, 24))
    np.testing.assert_array_equal(np.asarray(out[:, 8:, :]), 1)

  def test_layer14_engram_embed_fsdp_padded_slice_bit_identical(self):
    """Verifies Option A: engram.embed 620-row fsdp=4 padding produces bit-identical
    forward activations and backward active gradients."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "fsdp"))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", None))

    # 1. 618 active rows on a 1000-row table
    active_rows = np.arange(10, 628, dtype=np.int64)
    plan = lw_model.plan_engram_slice(
        active_row_ids=active_rows,
        control_row_ids=np.array([], dtype=np.int64),
        num_table_rows=1000,
        row_multiple=4,
    )
    self.assertEqual(len(plan.row_ids), 618)
    self.assertEqual(plan.num_slice_rows, 620)
    self.assertEqual(plan.poison_row, 618)
    self.assertEqual(plan.masked_pad_row, 619)

    rng = np.random.default_rng(1234)
    emb_dim = 256
    w_full = rng.standard_normal((1000, emb_dim)).astype(np.float32)
    w_unpadded = w_full[plan.row_ids].copy()

    # Cached weight array might have unpadded shape (618, 256)
    cpu_w = {"engram": {"embed": w_unpadded}}
    abstract_w = {
        "engram": {"embed": jax.ShapeDtypeStruct((plan.num_slice_rows, emb_dim), jnp.float32, sharding=sharding)}
    }

    # Proves (618, 256) raises IndivisibleError when placed on fsdp=4
    with self.assertRaises(Exception) as cm:
      jax.device_put(w_unpadded, sharding)
    self.assertEqual(type(cm.exception).__name__, "IndivisibleError")

    # Padding helper expands (618, 256) -> (620, 256)
    padded_cpu_w = lw_chain._pad_engram_weights_if_needed(cpu_w, abstract_w)  # pylint: disable=protected-access
    w_padded = padded_cpu_w["engram"]["embed"]
    self.assertEqual(w_padded.shape, (620, emb_dim))

    # Padded array satisfies P("fsdp", None) divisibility
    dev_w = jax.device_put(w_padded, sharding)
    self.assertEqual(dev_w.shape, (620, emb_dim))

    # 2. Forward pass bit-identical check
    token_slice_indices = rng.integers(0, len(plan.row_ids), size=(2, 16))
    out_padded = w_padded[token_slice_indices]
    out_unpadded = w_unpadded[token_slice_indices]
    np.testing.assert_array_equal(out_padded, out_unpadded)

    # 3. Backward pass (VJP) bit-identical check
    cotangent = rng.standard_normal(out_padded.shape).astype(np.float32)

    def loss_fn(w, idx):
      return jnp.sum(w[idx] * cotangent)

    grad_padded = jax.grad(loss_fn)(w_padded, token_slice_indices)
    grad_unpadded = jax.grad(loss_fn)(w_unpadded, token_slice_indices)

    # Padded rows 618..619 are identically 0.0
    self.assertTrue(np.all(grad_padded[618:] == 0.0))
    # Active rows 0..617 match unpadded reference gradient bit-identically
    np.testing.assert_array_equal(grad_padded[:618], grad_unpadded)

  def test_verify_layerwise_fast_preflight_fails_fast_on_stale_harness_sha256(self):
    """Verifies fast preflight check (< 2s) passes on valid AOT records and fails fast on stale harness SHA256."""
    t0 = time.time()
    valid_record = {
        "program": "L4.layer[20]",
        "tier": "production",
        "sample": "M4",
        "seq_len": 1024,
        "batch_size": 1,
        "num_images": 2,
        "vision_grid": [39, 39],
        "layers": [20],
        "engram_sliced": True,
        "activation_quant": False,
        "harness_code_sha256": lw_checks.harness_code_sha256(),
        "model_code_sha256": lw_checks.model_code_sha256(),
        "compiled": True,
        "fits_device_hbm": True,
    }
    stale_record = {
        **valid_record,
        "harness_code_sha256": "0" * 64,
    }
    aot_valid = {"programs": [valid_record]}
    aot_stale = {"programs": [stale_record]}

    # Valid record passes fast in milliseconds
    records = lw_stages.preflight_aot(
        stage="L4",
        tier="production",
        sample="M4",
        target="20",
        aot_json_data=aot_valid,
        num_devices=4,
    )
    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["program"], "L4.layer[20]")
    self.assertLess(time.time() - t0, 2.0)

    # Stale harness_code_sha256 fails fast with ValueError matching compiled from harness code
    with self.assertRaisesRegex(ValueError, "compiled from harness code 0000"):
      lw_stages.preflight_aot(
          stage="L4",
          tier="production",
          sample="M4",
          target="20",
          aot_json_data=aot_stale,
          num_devices=4,
      )

    # Child process test exercising verify_layerwise.py CLI flags and exit codes in < 2 seconds
    repo = pathlib.Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"}
    with tempfile.TemporaryDirectory() as tmp_dir:
      stale_file = pathlib.Path(tmp_dir) / "lw_aot_v5p_8_M4.json"
      stale_file.write_text(json.dumps(aot_stale), encoding="utf-8")
      t_proc = time.time()
      proc_fail = subprocess.run(
          [
              sys.executable,
              "-m",
              "tests.utils.deepseek_v41_layerwise.verify_layerwise",
              "--preflight-only",
              "--stage=L4",
              "--precision-tier=production",
              "--sample=M4",
              "--target=20",
              f"--aot-json={stale_file}",
          ],
          cwd=repo,
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      proc_elapsed = time.time() - t_proc
      self.assertNotEqual(proc_fail.returncode, 0)
      self.assertIn("compiled from harness code", proc_fail.stderr + proc_fail.stdout)
      self.assertLess(proc_elapsed, 20.0)

      valid_file = pathlib.Path(tmp_dir) / "lw_aot_v5p_8_M4_valid.json"
      valid_file.write_text(json.dumps(aot_valid), encoding="utf-8")
      proc_pass = subprocess.run(
          [
              sys.executable,
              "-m",
              "tests.utils.deepseek_v41_layerwise.verify_layerwise",
              "--preflight-only",
              "--stage=L4",
              "--precision-tier=production",
              "--sample=M4",
              "--target=20",
              f"--aot-json={valid_file}",
          ],
          cwd=repo,
          env=env,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(proc_pass.returncode, 0)
      self.assertIn("[PREFLIGHT] All 1 stage AOT records verified successfully", proc_pass.stdout)

  def test_weight_cache_dir_corrupt_or_sha256_mismatch_rejected_before_layer0(self):
    """Verifies that corrupt, empty, truncated .pkl or missing/mismatched .sha256 sidecars are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      cache_dir = pathlib.Path(tmp_dir)

      # 1. Non-existent cache dir raises ValueError
      non_existent = cache_dir / "non_existent"
      with self.assertRaisesRegex(ValueError, "Weight cache directory does not exist"):
        lw_chain.verify_weight_cache_dir(non_existent, num_layers=2)

      # 2. 0-byte cache file raises ValueError
      f0 = cache_dir / "layer_00.pkl"
      f0.touch()
      with self.assertRaisesRegex(ValueError, "Weight cache file is empty"):
        lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

      # 3. Corrupt pickle header raises ValueError
      f0.write_bytes(b"\x00\x00corrupt_data")
      with self.assertRaisesRegex(ValueError, "invalid or corrupted pickle header"):
        lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

      # 4. Valid pickle but missing .sha256 sidecar raises ValueError
      valid_data = pickle.dumps({"weights": [1.0, 2.0]}, protocol=pickle.HIGHEST_PROTOCOL)
      f0.write_bytes(valid_data)
      with self.assertRaisesRegex(ValueError, "missing sha256 sidecar"):
        lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

      # 5. Invalid sha256 sidecar format raises ValueError
      sha_file = cache_dir / "layer_00.pkl.sha256"
      sha_file.write_text("bad_hash\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Corrupt sha256 sidecar"):
        lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

      # 6. Mismatched sha256 checksum raises ValueError
      wrong_sha = "0" * 64
      sha_file.write_text(f"{wrong_sha}  layer_00.pkl\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Weight cache checksum mismatch"):
        lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

      # 7. Valid sha256 sidecar passes verification
      true_sha = hashlib.sha256(valid_data).hexdigest()
      sha_file.write_text(f"{true_sha}  layer_00.pkl\n", encoding="utf-8")
      lw_chain.verify_weight_cache_dir(cache_dir, num_layers=1)

  def test_get_or_cache_layer_cpu_weights_skips_engram(self):
    """Verifies that _get_or_cache_layer_cpu_weights bypasses static disk cache when engram_rows is set."""
    # pylint: disable=protected-access
    with tempfile.TemporaryDirectory() as tmp_dir:
      cache_dir = pathlib.Path(tmp_dir)
      cache_file = cache_dir / "layer_01.pkl"
      cache_file.write_bytes(pickle.dumps({"weights": [1.0, 2.0]}))

      mock_config = mock.MagicMock()
      mock_mesh = mock.MagicMock()
      mock_engram = mock.MagicMock(spec=lw_stages.EngramRows)
      skeletons = {(id(mock_mesh), id(mock_config), 1): ("mock_graphdef", "mock_rest", None, {})}

      with mock.patch.object(lw_stages, "load_layer") as mock_load:
        mock_parts = mock.MagicMock()
        mock_parts.weights = {"w": jax.ShapeDtypeStruct((2, 2), jnp.float32, sharding=None)}
        mock_parts.graphdef = "mock_gdef"
        mock_parts.rest = "mock_rest"
        mock_load.return_value = (mock_parts, None)

        # 1. With engram_rows is None: loads from cache_file without calling load_layer
        with mock.patch("pickle.load", return_value={"weights": [1.0, 2.0]}):
          _ = lw_chain._get_or_cache_layer_cpu_weights(
              1,
              mock_config,
              mock_mesh,
              pathlib.Path("/tmp"),
              engram_rows=None,
              weight_cache_dir=cache_dir,
              skeletons=skeletons,
          )
          self.assertEqual(mock_load.call_count, 0)

        # 2. With engram_rows is not None: bypasses cache_file and calls load_layer directly
        mock_load.reset_mock()
        _ = lw_chain._get_or_cache_layer_cpu_weights(
            1,
            mock_config,
            mock_mesh,
            pathlib.Path("/tmp"),
            engram_rows=mock_engram,
            weight_cache_dir=cache_dir,
            skeletons=skeletons,
        )
        self.assertEqual(mock_load.call_count, 1)

  def test_assert_replay_config_parity_positive_and_negative_controls(self):
    """Verifies that assert_replay_config_parity enforces R3 tier precision contract and rejects invalid configs."""
    # pylint: disable=import-outside-toplevel
    from tests.utils.deepseek_v41_layerwise import replay_r3_b16_v5p8

    class MockValidConfig:
      dtype = "float32"
      dtype_mm = "float32"
      matmul_precision = "highest"
      weight_dtype = "bfloat16"
      float32_gate_logits = True
      logical_axis_rules = [("data", "data")]

    valid_cfg = MockValidConfig()
    replay_r3_b16_v5p8.assert_replay_config_parity(valid_cfg)

    # Negative controls
    class MismatchedDtype(MockValidConfig):
      dtype = "bfloat16"

    with self.assertRaisesRegex(ValueError, "Replay config requires dtype='float32'"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedDtype())

    class MismatchedDtypeMm(MockValidConfig):
      dtype_mm = "bfloat16"

    with self.assertRaisesRegex(ValueError, "Replay config requires dtype_mm='float32'"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedDtypeMm())

    class MismatchedMatmulPrecision(MockValidConfig):
      matmul_precision = "default"

    with self.assertRaisesRegex(ValueError, "Replay config requires matmul_precision='highest'"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedMatmulPrecision())

    class MismatchedWeightDtype(MockValidConfig):
      weight_dtype = "float32"

    with self.assertRaisesRegex(ValueError, "Replay config requires weight_dtype='bfloat16'"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedWeightDtype())

    class MismatchedGateLogits(MockValidConfig):
      float32_gate_logits = False

    with self.assertRaisesRegex(ValueError, "Replay config requires float32_gate_logits=True"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedGateLogits())

    class MismatchedLogicalAxisRules(MockValidConfig):
      logical_axis_rules = []

    with self.assertRaisesRegex(ValueError, "Replay config requires non-empty logical_axis_rules"):
      replay_r3_b16_v5p8.assert_replay_config_parity(MismatchedLogicalAxisRules())

  def test_layer_parity_torch_free_compare_positive_and_negative_controls(self):
    """Verifies that _LayerParityTorchFree.compare computes metrics with zero-norm safeguard and enforces tolerances."""
    # pylint: disable=import-outside-toplevel,redefined-outer-name
    from tests.utils.deepseek_v41_layerwise.lw_checks import _LayerParityTorchFree, _Tolerance

    tol = _Tolerance(atol=1e-5, rtol=1e-5, frob_rel=1e-5)
    ref = np.ones((4, 4), dtype=np.float32) * 2.0
    cand = np.copy(ref)

    # 1. Positive identical match
    passed, metrics = _LayerParityTorchFree.compare(cand, ref, tol)
    self.assertTrue(passed)
    self.assertTrue(metrics.finite)
    self.assertEqual(metrics.max_abs, 0.0)
    self.assertEqual(metrics.mean_abs, 0.0)
    self.assertEqual(metrics.rmse, 0.0)
    self.assertAlmostEqual(metrics.cos, 1.0, places=5)
    self.assertEqual(metrics.frob_rel, 0.0)
    self.assertEqual(metrics.ulps, 0.0)

    # 2. Negative control: perturbed tensor exceeding tolerance
    perturbed = np.copy(ref)
    perturbed[0, 0] += 0.5
    passed_neg, metrics_neg = _LayerParityTorchFree.compare(perturbed, ref, tol)
    self.assertFalse(passed_neg)
    self.assertTrue(metrics_neg.finite)
    self.assertAlmostEqual(metrics_neg.max_abs, 0.5, places=5)
    self.assertGreater(metrics_neg.frob_rel, 1e-5)
    self.assertGreater(metrics_neg.ulps, 1.0)

    # 3. Negative control: norm < 1e-7 zero-norm safeguard
    tiny_ref = np.zeros((4, 4), dtype=np.float32)
    tiny_cand = np.zeros((4, 4), dtype=np.float32)
    passed_tiny, metrics_tiny = _LayerParityTorchFree.compare(tiny_cand, tiny_ref, tol)
    self.assertTrue(passed_tiny)
    self.assertEqual(metrics_tiny.cos, 0.0)

    # 4. Negative control: non-finite values return finite=False
    nan_cand = np.copy(ref)
    nan_cand[0, 0] = np.nan
    passed_nan, metrics_nan = _LayerParityTorchFree.compare(nan_cand, ref, tol)
    self.assertFalse(passed_nan)
    self.assertFalse(metrics_nan.finite)

    # 5. Invalid inputs raise
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      _LayerParityTorchFree.compare(np.ones((2, 2)), np.ones((3, 3)), tol)
    with self.assertRaisesRegex(TypeError, "requires floating-point arrays"):
      _LayerParityTorchFree.compare(np.ones((2, 2), dtype=np.int32), np.ones((2, 2)), tol)

  def test_verify_layerwise_execution_with_lw_runs_without_precision_tier(self):
    """Verifies that --lw-runs=L2:production:M4:1 populates precision_tier, stage, sample, target and executes."""
    # pylint: disable=import-outside-toplevel
    from tests.utils.deepseek_v41_layerwise import verify_layerwise

    argv = [
        "verify_layerwise",
        "--lw-runs=L2:production:M4:1",
        "--aot-dir=/tmp/fake_aot",
        "--output-json=/tmp/fake_out.json",
    ]
    with mock.patch("tests.utils.deepseek_v41_layerwise.verify_layerwise.run_preflight_checks") as mock_preflight:
      with mock.patch("tests.utils.deepseek_v41_layerwise.lw_runs.run_gate") as mock_run_gate:
        args = verify_layerwise.flags_parser(argv)
        verify_layerwise.main(args)

        mock_preflight.assert_called_once()
        mock_run_gate.assert_called_once()
        passed_args = mock_run_gate.call_args[0][0]
        self.assertEqual(passed_args.stage, "L2")
        self.assertEqual(passed_args.precision_tier, "production")
        self.assertEqual(passed_args.sample, "M4")
        self.assertEqual(passed_args.target, "1")
        self.assertEqual(passed_args.aot_json, "/tmp/fake_aot/lw_aot_v5p_8_M4.json")

    # Negative control: missing --output-json raises ValueError
    argv_no_out = ["verify_layerwise", "--lw-runs=L2:production:M4:1", "--aot-dir=/tmp/fake_aot"]
    with mock.patch("tests.utils.deepseek_v41_layerwise.verify_layerwise.run_preflight_checks"):
      args_no_out = verify_layerwise.flags_parser(argv_no_out)
      with self.assertRaisesRegex(ValueError, "Must provide --output-json"):
        verify_layerwise.main(args_no_out)


if __name__ == "__main__":

  unittest.main()

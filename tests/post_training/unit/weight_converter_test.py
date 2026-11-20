# Copyright 2026 Google LLC
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

"""Unit tests for the direct MaxText-to-MaxText weight converter (CPU-only).

Covers the qwen3.5 hybrid layout: an inhomogeneous scanned block of
`inhomogeneous_layer_cycle_interval` distinct layers on the trainer side,
unrolled to one attribute per layer on the rollout side, with the rollout
pre-fusing MoE `wi_0`/`wi_1` into a padded `wi`.
"""

import copy
import gc
import os
import resource

# Must precede the first JAX import: the cross-mesh tests below need more than
# one CPU device, and the backend reads this only at initialization.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from unittest import mock
import types as pytypes  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position
from flax import nnx, traverse_util  # pylint: disable=wrong-import-position

from maxtext.common.common_types import get_weight_dtype
from maxtext.integration.vllm import weight_converter
from maxtext.integration.vllm.weight_converter import (
    ConversionPlanError,
    MaxTextToMaxTextConverter,
    MoEFusedLayout,
    Rule,
    WeightConverter,
    MODEL_TO_CONVERSION_RULES,
    _group_plan,
)

pytestmark = pytest.mark.post_training

# Shrunk qwen3.5: 8 layers, cycle 4 -> 2 scanned blocks. Slots 0-2 are
# GatedDeltaNet, slot 3 is full attention, matching Qwen3_5DecoderLayer.
CYCLE = 4
NUM_LAYERS = 8
NUM_BLOCKS = NUM_LAYERS // CYCLE
SCAN_AXIS = 1
EMB = 8
EXPERTS = 4
MOE_DIM = 6


def _config(**overrides):
  """Returns a mock configuration object for tests."""
  cfg = pytypes.SimpleNamespace(
      num_decoder_layers=NUM_LAYERS,
      inhomogeneous_layer_cycle_interval=CYCLE,
      param_scan_axis=SCAN_AXIS,
      model_name="test_model",
      decoder_block="default",
      dtype="bfloat16",
      fp8_moe=False,
      rollout_fp8_moe=False,
      enable_dp_attention=False,
      kv_tp_size=1,
      moe_mlp_tp_size=1,
  )
  for key, value in overrides.items():
    setattr(cfg, key, value)
  return cfg


def _arr(*shape, offset=0):
  """Ramp of the given shape.

  `offset` shifts the whole ramp so two params of identical shape hold
  different values. Without it every cycle slot gets a byte-identical fixture
  and a test cannot tell "the converter mapped slot 1 correctly" apart from
  "the converter returned slot 0 twice".
  """
  size = int(np.prod(shape))
  return jnp.asarray((np.arange(size, dtype=np.float32) + float(offset)).reshape(shape))


def _scanned(*per_layer_shape, offset=0):
  """A trainer-side param with the scan dim inserted at `param_scan_axis`."""
  shape = list(per_layer_shape)
  shape.insert(SCAN_AXIS, NUM_BLOCKS)
  return _arr(*shape, offset=offset)


def _source_tree(fused_moe_on_target: bool):
  """Trainer state: scanned `layers.layer_{0..3}`, MoE always split."""
  layers = {}
  for slot in range(CYCLE):
    # Every slot gets its own value range, and gate/up differ within a slot, so
    # an incorrectly mapped slot or a swapped wi_0/wi_1 shows up as a value mismatch
    # rather than passing silently on identical fixtures.
    base = (slot + 1) * 1000
    block = {
        "input_layernorm": {"scale": _scanned(EMB, offset=base)},
        "moe_block": {
            "gate": {"kernel": _scanned(EMB, EXPERTS, offset=base + 100)},
            "wi_0": _scanned(EXPERTS, EMB, MOE_DIM, offset=base + 200),
            "wi_1": _scanned(EXPERTS, EMB, MOE_DIM, offset=base + 300),
            "wo": _scanned(EXPERTS, MOE_DIM, EMB, offset=base + 400),
        },
    }
    # Slot 3 is the full-attention layer; the others are linear (GDN).
    if slot == CYCLE - 1:
      block["self_attention"] = {"query": {"kernel": _scanned(EMB, 2, 4, offset=base + 500)}}
    else:
      block["linear_attention"] = {"conv": {"kernel": _scanned(EMB, 1, 4, offset=base + 500)}}
    layers[f"layer_{slot}"] = block
  del fused_moe_on_target
  return {
      "base": {
          "token_embedder": {"embedding": _arr(16, EMB)},
          "decoder": {"layers": layers, "decoder_norm": {"scale": _arr(EMB)}},
      }
  }


def _nested_source_tree(fused_moe_on_target: bool):
  """Trainer state in the nested-scan layout `Qwen3NextScannableBlock` produces."""
  legacy = _source_tree(fused_moe_on_target)["base"]["decoder"]["layers"]
  local_slots = [legacy[f"layer_{slot}"] for slot in range(CYCLE - 1)]
  tree = _source_tree(fused_moe_on_target)
  tree["base"]["decoder"]["layers"] = {
      "local_layers": jax.tree.map(lambda *slots: jnp.stack(slots, axis=SCAN_AXIS + 1), *local_slots),
      "global_layer": legacy[f"layer_{CYCLE - 1}"],
  }
  return tree


def _target_tree(moe_intermediate=MOE_DIM, fused=True, wo_intermediate=MOE_DIM):
  """Rollout state: unrolled `layers_{0..7}`, optionally fused MoE.

  `wo_intermediate` is separate from `moe_intermediate` because the rollout pads
  the two independently: `wi`'s padded dim is doubled by the gate/up fusion,
  `wo`'s is not.
  """
  decoder = {"decoder_norm": {"scale": _arr(EMB)}}
  for layer in range(NUM_LAYERS):
    slot = layer % CYCLE
    block = {
        "input_layernorm": {"scale": _arr(EMB)},
        "moe_block": {
            "gate": {"kernel": _arr(EMB, EXPERTS)},
            "wo": _arr(EXPERTS, wo_intermediate, EMB),
        },
    }
    if fused:
      block["moe_block"]["wi"] = _arr(EXPERTS, EMB, moe_intermediate * 2)
    else:
      block["moe_block"]["wi_0"] = _arr(EXPERTS, EMB, MOE_DIM)
      block["moe_block"]["wi_1"] = _arr(EXPERTS, EMB, MOE_DIM)
    if slot == CYCLE - 1:
      block["self_attention"] = {"query": {"kernel": _arr(EMB, 2, 4)}}
    else:
      block["linear_attention"] = {"conv": {"kernel": _arr(EMB, 1, 4)}}
    decoder[f"layers_{layer}"] = block
  return {
      "token_embedder": {"embedding": _arr(16, EMB)},
      "decoder": decoder,
  }


class ScannedToUnrolledMappingTest(unittest.TestCase):
  """The scanned/unrolled index map is the whole job; pin it down exactly."""

  def test_every_layer_maps_to_its_cycle_slot_and_block(self):
    converter = MaxTextToMaxTextConverter(_config())
    out = converter.convert(_source_tree(True), _target_tree())
    src = _source_tree(True)["base"]["decoder"]["layers"]

    for layer in range(NUM_LAYERS):
      slot, block = layer % CYCLE, layer // CYCLE
      got = out["decoder"][f"layers_{layer}"]["input_layernorm"]["scale"]
      want = jnp.take(src[f"layer_{slot}"]["input_layernorm"]["scale"], block, axis=SCAN_AXIS)
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_distinct_layers_within_a_cycle_are_not_confused(self):
    """layers_0 and layers_1 share a scan block but are different slots."""
    converter = MaxTextToMaxTextConverter(_config())
    out = converter.convert(_source_tree(True), _target_tree())["decoder"]
    self.assertFalse(
        np.array_equal(
            np.asarray(out["layers_0"]["input_layernorm"]["scale"]),
            np.asarray(out["layers_1"]["input_layernorm"]["scale"]),
        )
    )

  def test_hybrid_layer_types_both_transfer(self):
    """GDN slots and the full-attention slot have different subtrees."""
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), _target_tree())["decoder"]
    self.assertIn("conv", out["layers_0"]["linear_attention"])
    self.assertIn("query", out["layers_3"]["self_attention"])

  def test_non_layer_params_pass_through(self):
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), _target_tree())
    np.testing.assert_array_equal(
        np.asarray(out["token_embedder"]["embedding"]),
        np.asarray(_arr(16, EMB)),
    )

  def test_nested_scan_layout_converts_identically(self):
    """The nested-scan layout converts to the same rollout weights as the per-slot layout."""
    target = _target_tree()
    legacy = traverse_util.flatten_dict(MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), target))
    nested = traverse_util.flatten_dict(MaxTextToMaxTextConverter(_config()).convert(_nested_source_tree(True), target))
    self.assertEqual(set(legacy), set(nested))
    for key, want in legacy.items():
      np.testing.assert_array_equal(
          np.asarray(nested[key]),
          np.asarray(want),
          err_msg=f"{'.'.join(map(str, key))} differs between the per-slot and nested layouts",
      )

  def test_nested_local_layers_not_matching_the_cycle_are_rejected(self):
    """A `local_layers` axis must hold exactly the cycle's local slots, or slots would be dropped silently."""
    for num_slots in (CYCLE - 2, CYCLE):
      with self.subTest(num_slots=num_slots):
        source = _nested_source_tree(True)
        local = source["base"]["decoder"]["layers"]["local_layers"]
        local["input_layernorm"]["scale"] = jnp.take(
            local["input_layernorm"]["scale"], jnp.arange(num_slots) % (CYCLE - 1), axis=SCAN_AXIS + 1
        )
        with self.assertRaisesRegex(ConversionPlanError, "nested cycle-slot axis"):
          MaxTextToMaxTextConverter(_config()).convert(source, _target_tree())

  def test_nested_local_layers_cast_only_the_selected_slot(self):
    """Every slot's group reads the whole `local_layers` source, so the slot must be taken before the cast."""
    source = _nested_source_tree(True)
    local = traverse_util.flatten_dict(source["base"]["decoder"]["layers"]["local_layers"])
    slot_shapes = {
        ".".join(("decoder", "layers", "local_layers") + key): value.shape[: SCAN_AXIS + 1] + value.shape[SCAN_AXIS + 2 :]
        for key, value in local.items()
    }
    cast_shapes = []
    real_cast = weight_converter._apply_dtype_cast  # pylint: disable=protected-access

    def recording_cast(val, tgt_dtype, src_key):
      if "local_layers" in src_key:
        cast_shapes.append((src_key, val.shape))
      return real_cast(val, tgt_dtype, src_key)

    with mock.patch.object(weight_converter, "_apply_dtype_cast", side_effect=recording_cast):
      MaxTextToMaxTextConverter(_config()).convert(source, _target_tree())
    self.assertTrue(cast_shapes)
    for src_key, shape in cast_shapes:
      self.assertEqual(shape, slot_shapes[src_key], src_key)

  def test_plan_is_built_once_and_reused(self):
    converter = MaxTextToMaxTextConverter(_config())
    converter.convert(_source_tree(True), _target_tree())
    plan = converter._plan  # pylint: disable=protected-access
    converter.convert(_source_tree(True), _target_tree())
    self.assertIs(plan, converter._plan)  # pylint: disable=protected-access


class MoEFusionTest(unittest.TestCase):

  def test_split_source_fuses_into_target_wi(self):
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), _target_tree())
    wi = out["decoder"]["layers_0"]["moe_block"]["wi"]
    self.assertEqual(wi.shape, (EXPERTS, EMB, MOE_DIM * 2))

  def test_padded_target_intermediate_is_honoured(self):
    """The rollout pads the MoE intermediate dim for GMM_v2; shapes must match."""
    padded = 16
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), _target_tree(moe_intermediate=padded))
    self.assertEqual(
        out["decoder"]["layers_0"]["moe_block"]["wi"].shape,
        (EXPERTS, EMB, padded * 2),
    )

  def test_concat_layout_places_gate_before_up(self):
    converter = MaxTextToMaxTextConverter(_config(), moe_fused_layout=MoEFusedLayout.CONCAT)
    out = converter.convert(_source_tree(True), _target_tree())
    wi = np.asarray(out["decoder"]["layers_0"]["moe_block"]["wi"])
    src = _source_tree(True)["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    wi_0 = np.asarray(jnp.take(src["wi_0"], 0, axis=SCAN_AXIS))
    np.testing.assert_array_equal(wi[..., :MOE_DIM], wi_0)

  def test_padded_wo_is_zero_padded_not_repeated(self):
    """`wo`'s intermediate dim is its *contracting* axis.

    When the rollout pads it (qwen3.5 does; qwen3-30b happens not to), the pad
    must be zeros so the padded lanes contribute nothing to the output.
    Repeating the rows instead would double-count every padded lane, and for a
    pad that is not an integer multiple it cannot even be expressed -- which is
    how this was found: `_MOE_MLP_WEIGHTS` omitted `wo`, sending it down the
    repeat branch, where 16 % 6 != 0 raises ShapeMismatchError.
    """
    padded = 16
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), _target_tree(wo_intermediate=padded))
    wo = np.asarray(out["decoder"]["layers_0"]["moe_block"]["wo"])
    self.assertEqual(wo.shape, (EXPERTS, padded, EMB))

    src = _source_tree(True)["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    expected = np.asarray(jnp.take(src["wo"], 0, axis=SCAN_AXIS))
    np.testing.assert_array_equal(wo[:, :MOE_DIM, :], expected)
    np.testing.assert_array_equal(wo[:, MOE_DIM:, :], np.zeros((EXPERTS, padded - MOE_DIM, EMB)))

  def test_unfused_target_takes_wi_0_and_wi_1_directly(self):
    out = MaxTextToMaxTextConverter(_config()).convert(_source_tree(False), _target_tree(fused=False))
    moe = out["decoder"]["layers_0"]["moe_block"]
    self.assertIn("wi_0", moe)
    self.assertIn("wi_1", moe)
    self.assertNotIn("wi", moe)


class CoverageValidationTest(unittest.TestCase):
  """vLLM boots on dummy weights, so an unmatched target must not be silent."""

  def test_unmatched_target_param_raises(self):
    target = _target_tree()
    target["decoder"]["layers_0"]["mystery_block"] = {"kernel": _arr(EMB, EMB)}
    with self.assertRaises(ConversionPlanError) as ctx:
      MaxTextToMaxTextConverter(_config()).convert(_source_tree(True), target)
    self.assertIn("mystery_block", str(ctx.exception))

  def test_untransferred_source_param_raises(self):
    source = _source_tree(True)
    source["base"]["decoder"]["layers"]["layer_0"]["stray"] = {"kernel": _scanned(EMB)}
    with self.assertRaises(ConversionPlanError) as ctx:
      MaxTextToMaxTextConverter(_config()).convert(source, _target_tree())
    self.assertIn("stray", str(ctx.exception))

  def test_allowlisted_source_param_is_tolerated(self):
    source = _source_tree(True)
    source["base"]["vision_tower"] = {"proj": {"kernel": _arr(EMB, EMB)}}
    converter = MaxTextToMaxTextConverter(_config(), allow_unused_source_keys=("vision_tower",))
    converter.convert(source, _target_tree())  # must not raise

  def test_layer_count_not_divisible_by_cycle_raises(self):
    with self.assertRaises(ConversionPlanError):
      MaxTextToMaxTextConverter(_config(num_decoder_layers=NUM_LAYERS + 1))

  def test_wrong_scan_axis_is_reported(self):
    converter = MaxTextToMaxTextConverter(_config(param_scan_axis=0))
    with self.assertRaises(ConversionPlanError) as ctx:
      converter.convert(_source_tree(True), _target_tree())
    self.assertIn("scanned blocks", str(ctx.exception))


class WeightConverterModeDispatchTest(unittest.TestCase):
  """`WeightConverter` fronts both modes; the selector is `rules`."""

  def test_rules_none_runs_direct_maxtext_to_maxtext(self):
    converter = WeightConverter(rules=None, config=_config())
    out = converter.convert(_source_tree(True), target_state=_target_tree())
    self.assertIn("layers_0", out["decoder"])
    self.assertEqual(
        out["decoder"]["layers_0"]["moe_block"]["wi"].shape,
        (EXPERTS, EMB, MOE_DIM * 2),
    )

  def test_rules_none_requires_config(self):
    with self.assertRaises(ValueError):
      WeightConverter(rules=None)

  def test_empty_rule_list_is_rejected(self):
    """`rules=[]` would convert nothing and leave vLLM on dummy weights."""
    with self.assertRaises(ValueError) as ctx:
      WeightConverter(rules=[], config=_config())
    self.assertIn("dummy", str(ctx.exception))

  def test_rules_given_runs_torchax_renaming(self):
    converter = WeightConverter(
        rules=[Rule(["base.token_embedder.embedding"], "model.embed_tokens.weight")],
        tp=1,
    )
    out = converter.convert(_source_tree(True))
    np.testing.assert_array_equal(
        np.asarray(out["model"]["embed_tokens"]["weight"]),
        np.asarray(_arr(16, EMB)),
    )

  def test_rules_that_match_nothing_raise(self):
    converter = WeightConverter(rules=[Rule(["no.such.key"], "model.x")], tp=1)
    with self.assertRaises(ValueError) as ctx:
      converter.convert(_source_tree(True))
    self.assertIn("dummy weights", str(ctx.exception))

  def test_direct_only_models_have_no_hf_rule_table(self):
    """qwen3.5 is direct-path-only; its registry entry must be None, not []."""
    self.assertIsNone(MODEL_TO_CONVERSION_RULES["qwen35_moe"])
    self.assertTrue(MODEL_TO_CONVERSION_RULES["qwen3_moe"])


def _mesh(devices):
  return jax.sharding.Mesh(np.asarray(devices).reshape(2, 2), ("fsdp", "tensor"))


def _device_id_set(x):
  """Device ids backing an array, via its mesh."""
  return {int(d.id) for d in np.asarray(x.sharding.mesh.devices).flatten()}


def _place(tree, mesh, spec_for):
  """Commits every leaf of `tree` to `mesh` with the spec `spec_for` returns."""
  flat = traverse_util.flatten_dict(tree)
  return traverse_util.unflatten_dict(
      {key: jax.device_put(value, jax.sharding.NamedSharding(mesh, spec_for(key, value))) for key, value in flat.items()}
  )


@unittest.skipIf(
    jax.device_count() < 8,
    "needs 8 devices; set XLA_FLAGS=--xla_force_host_platform_device_count=8",
)
class CrossMeshPlacementTest(unittest.TestCase):
  """The converter must not let rollout-mesh devices into its computations.

  On a split Pathways cluster the trainer and rollout own disjoint halves of the
  device list, and the only legal mesh crossing is the resharding step that runs
  *after* conversion. If any converter op inherits its device assignment from
  the target instead of the source, the resulting executable spans both meshes
  and a worker dies with "ExecuteShard attempted to execute on device id N which
  is not addressable by this client" -- which on Pathways takes the whole
  JobSet down rather than raising a catchable Python error.

  These tests reproduce that topology on CPU: source on devices 0-3, target on
  devices 4-7.
  """

  def setUp(self):
    super().setUp()
    devices = jax.devices()
    self.src_mesh = _mesh(devices[:4])
    self.tgt_mesh = _mesh(devices[4:8])
    self.src_ids = {int(d.id) for d in devices[:4]}
    self.tgt_ids = {int(d.id) for d in devices[4:8]}

  @staticmethod
  def _target_spec(key, value):
    # Shard the fused MoE kernel so `_get_n_shards` reports >1 off the *target*
    # mesh -- the exact metadata read that must not drag the target mesh into
    # the computation. Everything else is replicated, which still commits the
    # array to its mesh.
    if key[-1] == "wi" and value.shape[-1] % 2 == 0:
      return jax.sharding.PartitionSpec(None, None, "tensor")
    return jax.sharding.PartitionSpec()

  def _converted(self, moe_intermediate=MOE_DIM, target_mesh=None):
    source = _place(_source_tree(True), self.src_mesh, lambda *_: jax.sharding.PartitionSpec())
    target = _place(
        _target_tree(moe_intermediate=moe_intermediate),
        target_mesh if target_mesh is not None else self.tgt_mesh,
        self._target_spec,
    )
    converter = MaxTextToMaxTextConverter(_config())
    return converter.convert(source, target), target

  def test_no_converted_weight_lands_on_the_target_mesh(self):
    out, _ = self._converted()
    leaked = {
        ".".join(map(str, key)): sorted(_device_id_set(value) & self.tgt_ids)
        for key, value in traverse_util.flatten_dict(out).items()
        if _device_id_set(value) & self.tgt_ids
    }
    self.assertEqual(leaked, {}, f"converted weights placed on rollout-mesh devices: {leaked}")

  def test_every_converted_weight_stays_on_the_source_mesh(self):
    out, _ = self._converted()
    for key, value in traverse_util.flatten_dict(out).items():
      self.assertTrue(
          _device_id_set(value) <= self.src_ids,
          f"{'.'.join(map(str, key))} spans devices "
          f"{sorted(_device_id_set(value))}, expected a subset of "
          f"{sorted(self.src_ids)}",
      )

  def test_splitting_the_meshes_does_not_change_the_values(self):
    """Only the target's *placement* differs; the layout math is held constant.

    Both runs shard the fused `wi` two ways, so `_get_n_shards` is 2 in each and
    the interleave layout is identical. Any difference is therefore placement
    leaking into the result rather than a legitimate layout change.
    """
    cross_mesh, _ = self._converted()
    same_mesh, _ = self._converted(target_mesh=self.src_mesh)
    cross_flat = traverse_util.flatten_dict(cross_mesh)
    for key, want in traverse_util.flatten_dict(same_mesh).items():
      np.testing.assert_array_equal(
          np.asarray(cross_flat[key]),
          np.asarray(want),
          err_msg=f"{'.'.join(map(str, key))} differs across meshes",
      )

  def test_padded_moe_fusion_stays_on_the_source_mesh(self):
    """The padding path reads pad amounts off the target sharding."""
    out, _ = self._converted(moe_intermediate=MOE_DIM + 2)
    for key, value in traverse_util.flatten_dict(out).items():
      self.assertTrue(
          _device_id_set(value) <= self.src_ids,
          f"{'.'.join(map(str, key))} leaked to {sorted(_device_id_set(value))}",
      )


def _profile_conversion_worker(is_streaming, result_queue):
  """Worker process to profile memory usage during conversion."""
  num_layers = 16
  cycle = 2
  scaled_emb = 128
  scaled_experts = 8
  scaled_mlp = 256
  blocks = num_layers // cycle

  cfg = pytypes.SimpleNamespace(
      kv_tp_size=1,
      moe_mlp_tp_size=1,
      dtype="bfloat16",
      fp8_moe=False,
      rollout_fp8_moe=False,
      inhomogeneous_layer_cycle_interval=cycle,
      num_decoder_layers=num_layers,
      param_scan_axis=1,
      padded_base_moe_mlp_dim=scaled_mlp,
      prefuse_moe_weights=True,
      weight_dtype=jnp.float32,
  )

  def _arr(*shape):  # pylint: disable=redefined-outer-name
    return jnp.ones(shape, dtype=jnp.float32)

  layers = {}
  for slot in range(cycle):
    layers[f"layer_{slot}"] = {
        "input_layernorm": {"scale": _arr(scaled_emb, blocks)},
        "post_self_attention_layernorm": {"scale": _arr(scaled_emb, blocks)},
        "self_attention": {
            "query": {"kernel": _arr(scaled_emb, blocks, 4, 32)},
            "key": {"kernel": _arr(scaled_emb, blocks, 2, 32)},
            "value": {"kernel": _arr(scaled_emb, blocks, 2, 32)},
            "out": {"kernel": _arr(4, blocks, 32, scaled_emb)},
        },
        "moe_block": {
            "gate": {"kernel": _arr(scaled_emb, blocks, scaled_experts)},
            "wi_0": _arr(scaled_experts, blocks, scaled_emb, scaled_mlp),
            "wi_1": _arr(scaled_experts, blocks, scaled_emb, scaled_mlp),
            "wo": _arr(scaled_experts, blocks, scaled_mlp, scaled_emb),
        },
    }
  scaled_source = {
      "base": {
          "token_embedder": {"embedding": _arr(256, scaled_emb)},
          "decoder": {"decoder_norm": {"scale": _arr(scaled_emb)}, "layers": layers},
      }
  }

  converter = MaxTextToMaxTextConverter(cfg, prefuse_moe_weights=True)
  gc.collect()
  before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

  if is_streaming:
    for piece in converter.convert_streaming(scaled_source, target_state=None, groups_per_piece=1):
      del piece
      gc.collect()
  else:
    out = converter.convert(scaled_source, target_state=None)
    del out
    gc.collect()

  after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  result_queue.put(after_rss - before_rss)


class TargetFreeConversionTest(unittest.TestCase):
  """Comprehensive test suite for target-free key synthesis and execution."""

  def test_case_0_raiden_unscan_fails_on_hybrid_cycle(self):
    from maxtext.integration.tunix.weight_mapping import raiden_unscan  # pylint: disable=import-outside-toplevel

    source = _source_tree(True)
    with self.assertRaises(ValueError) as ctx:
      raiden_unscan.unscan_layers(source, num_layers=NUM_LAYERS, scan_axis=SCAN_AXIS)
    self.assertIn("expected axis 1 to be 8 (num_layers=8, cycle_interval=1)", str(ctx.exception))

  def test_case_1_homogeneous_target_free_unroll(self):
    cfg = _config(inhomogeneous_layer_cycle_interval=1, num_decoder_layers=4)
    source = {
        "base": {
            "token_embedder": {"embedding": _arr(16, EMB)},
            "decoder": {
                "decoder_norm": {"scale": _arr(EMB)},
                "layers": {
                    "input_layernorm": {"scale": _arr(EMB, 4)},
                    "self_attention": {"query": {"kernel": _arr(EMB, 4, 2, 4)}},
                },
            },
        }
    }
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    self.assertIn("token_embedder", out_root)
    self.assertIn("decoder", out_root)
    for i in range(4):
      layer_key = f"layers_{i}"
      self.assertIn(layer_key, out_root["decoder"])
      scale = getattr(
          out_root["decoder"][layer_key]["input_layernorm"]["scale"],
          "value",
          out_root["decoder"][layer_key]["input_layernorm"]["scale"],
      )
      query = getattr(
          out_root["decoder"][layer_key]["self_attention"]["query"]["kernel"],
          "value",
          out_root["decoder"][layer_key]["self_attention"]["query"]["kernel"],
      )
      self.assertEqual(scale.shape, (EMB,))
      self.assertEqual(query.shape, (EMB, 2, 4))

  def test_case_2_hybrid_cycle_target_free_unroll(self):
    cfg = _config()
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    src_layers = source["base"]["decoder"]["layers"]
    for layer in range(NUM_LAYERS):
      slot, block = layer % CYCLE, layer // CYCLE
      got = getattr(
          out_root["decoder"][f"layers_{layer}"]["input_layernorm"]["scale"],
          "value",
          out_root["decoder"][f"layers_{layer}"]["input_layernorm"]["scale"],
      )
      want = jnp.take(src_layers[f"layer_{slot}"]["input_layernorm"]["scale"], block, axis=SCAN_AXIS)
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_case_3_prefused_moe_target_free(self):
    from maxtext.integration.vllm.convert_utils import compute_padded_moe_mlp_dim  # pylint: disable=import-outside-toplevel

    # Verify helper across topologies
    self.assertEqual(compute_padded_moe_mlp_dim(512, 2, 128), 512)
    self.assertEqual(compute_padded_moe_mlp_dim(512, 4, 128), 1024)
    self.assertEqual(compute_padded_moe_mlp_dim(512, 8, 128), 2048)

    # Verify target-free prefused MoE with padded dim
    padded_dim = 16
    cfg = _config(padded_base_moe_mlp_dim=padded_dim, prefuse_moe_weights=True)
    source = _source_tree(True)
    converter = MaxTextToMaxTextConverter(cfg, prefuse_moe_weights=True)
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    wi = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wi"], "value", out_root["decoder"]["layers_0"]["moe_block"]["wi"]
    )
    wo = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wo"], "value", out_root["decoder"]["layers_0"]["moe_block"]["wo"]
    )
    self.assertEqual(wi.shape, (EXPERTS, EMB, padded_dim * 2))
    self.assertEqual(wo.shape, (EXPERTS, padded_dim, EMB))

  def test_case_4_abstract_evaluation(self):
    cfg = _config(padded_base_moe_mlp_dim=16, prefuse_moe_weights=True)

    def to_struct(x):
      arr = getattr(x, "value", x)
      return jax.ShapeDtypeStruct(arr.shape, arr.dtype)

    abstract_source = jax.tree_util.tree_map(to_struct, _source_tree(True))
    converter = MaxTextToMaxTextConverter(cfg, prefuse_moe_weights=True)
    out = converter.convert(abstract_source, target_state=None)
    out_root = out["base"] if "base" in out else out
    for leaf in jax.tree_util.tree_leaves(out):
      val = getattr(leaf, "value", leaf)
      self.assertIsInstance(val, jax.ShapeDtypeStruct)
    wi = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wi"], "value", out_root["decoder"]["layers_0"]["moe_block"]["wi"]
    )
    self.assertEqual(wi.shape, (EXPERTS, EMB, 32))

  def test_case_5_parity_vs_raiden_unscan_on_homogeneous(self):
    from maxtext.integration.tunix.weight_mapping import raiden_unscan  # pylint: disable=import-outside-toplevel

    cfg = pytypes.SimpleNamespace(
        kv_tp_size=1,
        moe_mlp_tp_size=1,
        dtype="bfloat16",
        fp8_moe=False,
        rollout_fp8_moe=False,
        num_decoder_layers=4,
        inhomogeneous_layer_cycle_interval=1,
        param_scan_axis=1,
        weight_dtype=jnp.bfloat16,
        prefuse_moe_weights=False,
    )
    raw_source = {
        "token_embedder": {"embedding": _arr(16, EMB)},
        "decoder": {
            "decoder_norm": {"scale": _arr(EMB)},
            "layers": {
                "input_layernorm": {"scale": _arr(EMB, 4)},
                "self_attention": {"query": {"kernel": _arr(EMB, 4, 2, 4)}},
            },
        },
    }
    bf16_source = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        raw_source,
    )
    baseline_out = raiden_unscan.unscan_layers(bf16_source, num_layers=4, scan_axis=1)

    converter = MaxTextToMaxTextConverter(cfg, prefuse_moe_weights=False, target_dtype=jnp.bfloat16)
    converter_out = converter.convert(raw_source, target_state=None)

    base_flat = traverse_util.flatten_dict(baseline_out)
    conv_flat = traverse_util.flatten_dict(converter_out)

    self.assertEqual(set(base_flat.keys()), set(conv_flat.keys()))
    for k, base_val in base_flat.items():
      v_base = getattr(base_val, "value", base_val)
      v_conv = getattr(conv_flat[k], "value", conv_flat[k])
      self.assertEqual(v_base.shape, v_conv.shape, f"Shape mismatch at {k}")
      self.assertEqual(v_base.dtype, v_conv.dtype, f"Dtype mismatch at {k}")
      np.testing.assert_array_equal(np.asarray(v_base), np.asarray(v_conv), err_msg=f"Value mismatch at {k}")

  def test_case_6_streaming_piece_count_and_parity(self):
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    pieces = list(converter.convert_streaming(source, target_state=None, groups_per_piece=1))
    self.assertEqual(len(pieces), len(converter._direct._groups))  # pylint: disable=protected-access

    # Parity check against fresh non-streaming converter
    fresh_converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    expected_out = fresh_converter.convert(source, target_state=None)

    merged_flat = {}
    for piece in pieces:
      piece_flat = traverse_util.flatten_dict(piece)
      for k, v in piece_flat.items():
        self.assertNotIn(k, merged_flat, f"Duplicate key across pieces: {k}")
        merged_flat[k] = v

    expected_flat = traverse_util.flatten_dict(expected_out)
    self.assertEqual(set(merged_flat.keys()), set(expected_flat.keys()))
    for k, exp_val in expected_flat.items():
      v_exp = getattr(exp_val, "value", exp_val)
      v_got = getattr(merged_flat[k], "value", merged_flat[k])
      self.assertEqual(v_exp.shape, v_got.shape, f"Shape mismatch at {k}")
      self.assertEqual(v_exp.dtype, v_got.dtype, f"Dtype mismatch at {k}")
      np.testing.assert_array_equal(np.asarray(v_exp), np.asarray(v_got), err_msg=f"Value mismatch at {k}")

  def test_case_7_streaming_piece_batching(self):
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    pieces = list(converter.convert_streaming(source, target_state=None, groups_per_piece=2))
    num_groups = len(converter._direct._groups)  # pylint: disable=protected-access
    expected_piece_count = (num_groups + 1) // 2
    self.assertEqual(len(pieces), expected_piece_count)

    fresh_converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    expected_out = fresh_converter.convert(source, target_state=None)
    expected_flat = traverse_util.flatten_dict(expected_out)

    merged_flat = {}
    for piece in pieces:
      piece_flat = traverse_util.flatten_dict(piece)
      for k, v in piece_flat.items():
        self.assertNotIn(k, merged_flat, f"Duplicate key across pieces: {k}")
        merged_flat[k] = v

    self.assertEqual(set(merged_flat.keys()), set(expected_flat.keys()))
    for k, exp_val in expected_flat.items():
      v_exp = getattr(exp_val, "value", exp_val)
      v_got = getattr(merged_flat[k], "value", merged_flat[k])
      np.testing.assert_array_equal(np.asarray(v_exp), np.asarray(v_got))

  def test_case_8_weight_converter_convert_streaming_dispatch(self):
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    # Direct MaxText mode delegates correctly
    direct_wc = WeightConverter(config=cfg, rollout_backend="maxtext")
    pieces = list(direct_wc.convert_streaming(source, target_state=None))
    self.assertGreater(len(pieces), 0)

    # Torchax rules mode raises NotImplementedError
    rule = Rule(source_patterns=["some_pattern"], target_pattern="some_target")
    torchax_wc = WeightConverter(rules=[rule], rollout_backend="torchax")
    with self.assertRaises(NotImplementedError):
      list(torchax_wc.convert_streaming(source))

  def test_case_9_target_free_kv_head_replication(self):
    cfg = _config(
        base_num_kv_heads=2,
        inhomogeneous_layer_cycle_interval=1,
        num_decoder_layers=2,
    )
    key_val = _scanned(EMB, 2, 4)
    value_val = _scanned(EMB, 2, 4)
    source = {
        "base": {
            "decoder": {
                "layers": {
                    "self_attention": {
                        "key": {"kernel": key_val},
                        "value": {"kernel": value_val},
                    }
                }
            }
        }
    }

    # 1. Successful replication: kv_tp_size=4, base_num_kv_heads=2 -> kv_replication=2
    converter = WeightConverter(
        config=cfg,
        kv_tp_size=4,
        rollout_backend="maxtext",
    )
    self.assertEqual(converter._direct.kv_replication, 2)  # pylint: disable=protected-access
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    for layer_idx in range(2):
      layer_key = f"layers_{layer_idx}"
      key_kernel = getattr(
          out_root["decoder"][layer_key]["self_attention"]["key"]["kernel"],
          "value",
          out_root["decoder"][layer_key]["self_attention"]["key"]["kernel"],
      )
      self.assertEqual(key_kernel.shape, (EMB, 4, 4))
      # Heads 0 and 1 are repeated along axis -2
      np.testing.assert_array_equal(key_kernel[:, 0, :], key_kernel[:, 1, :])
      np.testing.assert_array_equal(key_kernel[:, 2, :], key_kernel[:, 3, :])

    # 2. Divisibility failure: kv_tp_size=3, base_num_kv_heads=2
    with self.assertRaises(ValueError) as ctx:
      WeightConverter(
          config=cfg,
          kv_tp_size=3,
          rollout_backend="maxtext",
      )
    self.assertIn("must be divisible by base_num_kv_heads", str(ctx.exception))

  def test_case_10_target_free_dtype_cast_f32_to_bf16_all_param_kinds(self):
    cfg = _config(
        base_num_kv_heads=2,
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
        weight_dtype=jnp.float32,
        rollout_weight_dtype=jnp.bfloat16,
    )
    source = _source_tree(True)
    # Add an unscanned router/gate param, a KV kernel for replication, and a non-floating leaf.
    source["base"]["decoder"]["layers"]["layer_3"]["self_attention"]["key"] = {"kernel": _scanned(EMB, 2, 4, offset=4600)}
    source["base"]["decoder"]["router_gate"] = {
        "kernel": _arr(EMB, EXPERTS, offset=9000),
        "step_count": jnp.asarray([1, 2], dtype=jnp.int32),
    }

    converter = WeightConverter(
        config=cfg,
        kv_tp_size=4,
        rollout_backend="maxtext",
    )
    out = converter.convert(source, target_state=None)
    flat_out = traverse_util.flatten_dict(out)

    # Check MoE gate kernel, dense query kernel, norm scales, fused wi, wo, and KV-replicated key kernel.
    expected_bf16_paths = [
        ("base", "token_embedder", "embedding"),
        ("base", "decoder", "decoder_norm", "scale"),
        ("base", "decoder", "router_gate", "kernel"),
        ("base", "decoder", "layers_0", "input_layernorm", "scale"),
        ("base", "decoder", "layers_0", "moe_block", "gate", "kernel"),
        ("base", "decoder", "layers_0", "moe_block", "wi"),
        ("base", "decoder", "layers_0", "moe_block", "wo"),
        ("base", "decoder", "layers_3", "self_attention", "query", "kernel"),
        ("base", "decoder", "layers_3", "self_attention", "key", "kernel"),
    ]
    for path in expected_bf16_paths:
      self.assertIn(path, flat_out, f"Missing expected converted path: {path}")
      leaf = getattr(flat_out[path], "value", flat_out[path])
      self.assertEqual(leaf.dtype, jnp.bfloat16, f"Expected bfloat16 at {path}, got {leaf.dtype}")

    # Every floating leaf across the entire output must be bfloat16; integer leaves stay int32.
    for path, param in flat_out.items():
      leaf = getattr(param, "value", param)
      if path[-1] == "step_count":
        self.assertEqual(leaf.dtype, jnp.int32)
      else:
        self.assertEqual(leaf.dtype, jnp.bfloat16, f"Expected bfloat16 at {path}, got {leaf.dtype}")

  def test_case_11_target_free_rejects_fp8_target_dtype(self):
    source = _source_tree(True)
    for bad_dtype in ("float8_e4m3fn", jnp.float8_e4m3fn, "float8_e5m2", jnp.float8_e5m2):
      with self.subTest(bad_dtype=bad_dtype):
        cfg = _config(
            inhomogeneous_layer_cycle_interval=CYCLE,
            num_decoder_layers=NUM_LAYERS,
            prefuse_moe_weights=True,
            weight_dtype=jnp.float32,
            rollout_weight_dtype=bad_dtype,
        )
        converter = WeightConverter(config=cfg, rollout_backend="maxtext")
        with self.assertRaisesRegex(ConversionPlanError, "Unsupported target_dtype"):
          converter.convert(source, target_state=None)

  def test_case_12_target_free_dtype_matches_rollout_per_module_rule(self):
    """Each module is emitted in the dtype the rollout model allocates it in.

    Raiden's preflight rejects any leaf whose item size differs from the rollout
    buffer. The rollout allocates parameters via `get_weight_dtype(config, module)`
    (decoder_norm and the attention q/k norms read `config.weight_dtype`
    directly), which today returns `config.weight_dtype` for every non-FP8
    config, so one cast to `rollout_weight_dtype` covers every module. The rollout
    config sets `float32_gate_logits=True` and the tree includes the modules that
    branches keyed on that flag store in float32 (MoE gate, norms, GDN
    parameters, shared_expert_gate, logits_dense): if `get_weight_dtype` gains
    such a per-module rule, this test fails until the converter mirrors it.
    """
    rollout_cfg = pytypes.SimpleNamespace(
        kv_tp_size=1,
        moe_mlp_tp_size=1,
        fp8_moe=False,
        rollout_fp8_moe=False,
        weight_dtype=jnp.dtype(jnp.bfloat16),
        dtype=jnp.dtype(jnp.bfloat16),
        float32_gate_logits=True,
    )
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
        weight_dtype=jnp.float32,
        rollout_weight_dtype=rollout_cfg.weight_dtype,
    )
    source = _source_tree(True)
    decoder = source["base"]["decoder"]
    gdn_layer = decoder["layers"]["layer_0"]
    gdn_layer["post_attention_layernorm"] = {"scale": _scanned(EMB, offset=5000)}
    gdn_layer["moe_block"]["gate"]["bias"] = _scanned(EXPERTS, offset=5100)
    gdn_layer["moe_block"]["shared_expert_gate"] = {"kernel": _scanned(EMB, 1, offset=5200)}
    gdn_layer["linear_attention"].update(
        {
            "in_proj_ba": {"kernel": _scanned(EMB, 4, offset=5300)},
            "conv1d": {"kernel": _scanned(4, 1, 8, offset=5400)},
            "A_log": _scanned(2, offset=5500),
            "dt_bias": _scanned(2, offset=5600),
            "norm": {"scale": _scanned(4, offset=5700)},
        }
    )
    attn_slot = CYCLE - 1
    attention = decoder["layers"][f"layer_{attn_slot}"]["self_attention"]
    attention["query_norm"] = {"scale": _scanned(4, offset=5800)}
    attention["key_norm"] = {"scale": _scanned(4, offset=5900)}
    decoder["logits_dense"] = {"kernel": _arr(EMB, 16, offset=6000)}

    out = WeightConverter(config=cfg, rollout_backend="maxtext").convert(source, target_state=None)
    flat_out = traverse_util.flatten_dict(out)

    # Output path -> module name the rollout passes to `get_weight_dtype`; None means
    # the rollout reads `config.weight_dtype` directly.
    expected = {
        ("base", "token_embedder", "embedding"): "token_embedder",
        ("base", "decoder", "decoder_norm", "scale"): None,
        ("base", "decoder", "logits_dense", "kernel"): "logits_dense",
    }
    per_slot = {
        0: {
            ("input_layernorm", "scale"): "norm",
            ("post_attention_layernorm", "scale"): "norm",
            ("moe_block", "gate", "kernel"): "gate",
            ("moe_block", "gate", "bias"): "gate",
            ("moe_block", "shared_expert_gate", "kernel"): "shared_expert_gate",
            ("linear_attention", "in_proj_ba", "kernel"): "in_proj_ba",
            ("linear_attention", "conv1d", "kernel"): "conv1d",
            ("linear_attention", "A_log"): "A_log",
            ("linear_attention", "dt_bias"): "dt_bias",
            ("linear_attention", "norm", "scale"): "norm",
        },
        attn_slot: {
            ("self_attention", "query_norm", "scale"): None,
            ("self_attention", "key_norm", "scale"): None,
        },
    }
    for block in range(NUM_BLOCKS):
      for slot, modules in per_slot.items():
        layer_key = ("base", "decoder", f"layers_{block * CYCLE + slot}")
        expected.update({layer_key + suffix: module for suffix, module in modules.items()})

    for path, module in expected.items():
      with self.subTest(path=".".join(path)):
        self.assertIn(path, flat_out)
        want = rollout_cfg.weight_dtype if module is None else get_weight_dtype(rollout_cfg, module)
        leaf = getattr(flat_out[path], "value", flat_out[path])
        self.assertEqual(leaf.dtype, jnp.dtype(want))

  def test_case_13_dcp_moe_fusion_uses_single_shard(self):
    """Under DCP (tp=4, kv_tp_size=4, moe_mlp_tp_size=1), MoE wi must fuse with n_shards=1."""
    cfg = _config(prefuse_moe_weights=True)
    source = _source_tree(True)
    src_moe = source["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    expected_wi_0 = np.asarray(jnp.take(src_moe["wi_0"], 0, axis=SCAN_AXIS))
    expected_wi_1 = np.asarray(jnp.take(src_moe["wi_1"], 0, axis=SCAN_AXIS))
    expected_concat = np.concatenate([expected_wi_0, expected_wi_1], axis=-1)

    # 1. Target-free DCP conversion (kv_tp_size=4, moe_mlp_tp_size=1 resolved from config when omitted)
    converter = WeightConverter(
        config=_config(prefuse_moe_weights=True, kv_tp_size=4, moe_mlp_tp_size=1),
        tp=4,
        prefuse_moe_weights=True,
        rollout_backend="maxtext",
    )
    self.assertEqual(converter.kv_tp_size, 4)
    self.assertEqual(converter.moe_mlp_tp_size, 1)
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    wi_tf = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
        "value",
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
    )
    np.testing.assert_array_equal(np.asarray(wi_tf), expected_concat)

    # 2. Target-guided DCP conversion where target wi is expert-sharded over dcp on axis 0
    if jax.device_count() >= 4:
      dcp_mesh = jax.sharding.Mesh(
          np.asarray(jax.devices()[:4]).reshape(1, 2, 2),
          ("attn_dp", "model", "dcp"),
      )
      target = _place(
          _target_tree(),
          dcp_mesh,
          lambda key, val: (
              jax.sharding.PartitionSpec("dcp", None, None) if key[-1] == "wi" else jax.sharding.PartitionSpec()
          ),
      )
      guided_converter = WeightConverter(rules=None, config=cfg, tp=4)
      guided_out = guided_converter.convert(source, target_state=target)
      wi_guided = guided_out["decoder"]["layers_0"]["moe_block"]["wi"]
      np.testing.assert_array_equal(np.asarray(wi_guided), expected_concat)

  def test_case_14_nested_scan_layout_target_free(self):
    """Target-free conversion reads the nested `Qwen3NextScannableBlock` layout to the per-slot answer.

    One `local_layers` source feeds a plan group per cycle slot, so it must stay
    available until the last of those groups has run.
    """
    cfg = _config(padded_base_moe_mlp_dim=MOE_DIM + 2, prefuse_moe_weights=True)

    def convert(source, streaming):
      converter = MaxTextToMaxTextConverter(cfg, prefuse_moe_weights=True)
      if streaming:
        flat = {}
        for piece in converter.convert_streaming(source, target_state=None, groups_per_piece=1):
          flat.update(traverse_util.flatten_dict(piece))
        return flat
      return traverse_util.flatten_dict(converter.convert(source, target_state=None))

    for streaming in (False, True):
      with self.subTest(streaming=streaming):
        legacy = convert(_source_tree(True), streaming)
        nested = convert(_nested_source_tree(True), streaming)
        self.assertEqual(set(legacy), set(nested))
        for key, want in legacy.items():
          np.testing.assert_array_equal(
              np.asarray(getattr(nested[key], "value", nested[key])),
              np.asarray(getattr(want, "value", want)),
              err_msg=f"{'.'.join(map(str, key))} differs between the per-slot and nested layouts",
          )

    def to_struct(x):
      return jax.ShapeDtypeStruct(x.shape, x.dtype)

    abstract = {
        key: getattr(value, "value", value).shape
        for key, value in convert(jax.tree_util.tree_map(to_struct, _nested_source_tree(True)), False).items()
    }
    self.assertEqual(abstract, {key: getattr(value, "value", value).shape for key, value in legacy.items()})

  def test_case_10_moe_fp8_weight_sync(self):
    """Verifies BF16 trainer -> MoE FP8 sampler weight sync with per-channel scales."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    target = _target_tree(fused=True)
    # Convert target MoE layers to float8 and add scale tensors
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    out = converter.convert(source, target_state=target)

    for layer in range(NUM_LAYERS):
      moe_out = out["decoder"][f"layers_{layer}"]["moe_block"]
      wi = getattr(moe_out["wi"], "value", moe_out["wi"])
      wi_scale = getattr(moe_out["wi_scale"], "value", moe_out["wi_scale"])
      wo = getattr(moe_out["wo"], "value", moe_out["wo"])
      wo_scale = getattr(moe_out["wo_scale"], "value", moe_out["wo_scale"])

      self.assertEqual(wi.dtype, jnp.float8_e4m3fn)
      self.assertEqual(wi.shape, (EXPERTS, EMB, MOE_DIM * 2))
      self.assertEqual(wi_scale.dtype, jnp.float32)
      self.assertEqual(wi_scale.shape, (EXPERTS, 1, MOE_DIM * 2))

      self.assertEqual(wo.dtype, jnp.float8_e4m3fn)
      self.assertEqual(wo.shape, (EXPERTS, MOE_DIM, EMB))
      self.assertEqual(wo_scale.dtype, jnp.float32)
      self.assertEqual(wo_scale.shape, (EXPERTS, 1, EMB))

      # Scales should be positive non-zero floats
      self.assertTrue(bool(jnp.all(wi_scale > 0)))
      self.assertTrue(bool(jnp.all(wo_scale > 0)))

  def test_case_11_moe_fp8_weight_sync_unfused(self):
    """prefuse_moe_weights=false: the rollout keeps wi_0/wi_1, and each gets its own per-channel scale."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=False,
    )
    source = _source_tree(True)
    target = _target_tree(fused=False)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      for name, out_dim in (("wi_0", MOE_DIM), ("wi_1", MOE_DIM), ("wo", EMB)):
        moe[name] = moe[name].astype(jnp.float8_e4m3fn)
        moe[f"{name}_scale"] = jnp.ones((EXPERTS, 1, out_dim), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    out = converter.convert(source, target_state=target)

    for layer in range(NUM_LAYERS):
      slot, block = layer % CYCLE, layer // CYCLE
      src_moe = source["base"]["decoder"]["layers"][f"layer_{slot}"]["moe_block"]
      moe_out = out["decoder"][f"layers_{layer}"]["moe_block"]
      for name, out_dim in (("wi_0", MOE_DIM), ("wi_1", MOE_DIM), ("wo", EMB)):
        q = getattr(moe_out[name], "value", moe_out[name])
        scale = getattr(moe_out[f"{name}_scale"], "value", moe_out[f"{name}_scale"])
        self.assertEqual(q.dtype, jnp.float8_e4m3fn, name)
        self.assertEqual(scale.shape, (EXPERTS, 1, out_dim), name)
        want = jnp.take(src_moe[name], block, axis=SCAN_AXIS).astype(jnp.float32)
        got = q.astype(jnp.float32) * scale
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0.07, atol=0, err_msg=name)

  def test_case_12_fp8_scale_target_without_fp8_moe_raises(self):
    """A rollout with FP8 scales but a converter that does not quantize fails at plan time."""
    cfg = _config(inhomogeneous_layer_cycle_interval=CYCLE, num_decoder_layers=NUM_LAYERS, prefuse_moe_weights=True)
    target = _target_tree(fused=True)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=False)
    with self.assertRaisesRegex(weight_converter.ConversionPlanError, "FP8 scale"):
      converter.convert(_source_tree(True), target_state=target)

  def test_case_13_moe_fp8_scale_target_listed_first(self):
    """An FP8 group whose first target is the (E, 1, out) scale still aligns the weight to the weight target."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    target = _target_tree(fused=True)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    want = converter.convert(_source_tree(True), target_state=target)

    # Targets are sorted by slice index only, so reversing the plan puts each scale ahead of its weight.
    direct = converter._direct  # pylint: disable=protected-access
    direct._plan = list(reversed(direct._plan))  # pylint: disable=protected-access
    src_flat = traverse_util.flatten_dict(weight_converter._to_pure_dict(_source_tree(True)))  # pylint: disable=protected-access
    src_flat = {k[1:]: v for k, v in src_flat.items() if k and k[0] == "base"}
    direct._groups = _group_plan(direct._plan, fp8_moe=True, src_flat=src_flat)  # pylint: disable=protected-access
    fp8_groups = [g for g in direct._groups if g.quantize_to_fp8]  # pylint: disable=protected-access
    self.assertTrue(fp8_groups)
    self.assertTrue(all(str(g.targets[0][1][-1]).endswith("_scale") for g in fp8_groups))
    got = converter.convert(_source_tree(True), target_state=target)

    for layer in range(NUM_LAYERS):
      for name in ("wi", "wi_scale", "wo", "wo_scale"):
        g = got["decoder"][f"layers_{layer}"]["moe_block"][name]
        w = want["decoder"][f"layers_{layer}"]["moe_block"][name]
        g, w = getattr(g, "value", g), getattr(w, "value", w)
        np.testing.assert_array_equal(np.asarray(g, np.float32), np.asarray(w, np.float32), err_msg=name)

  def test_case_14_moe_fp8_quantizes_each_block_once(self):
    """A block feeds both a weight and a scale target; it is quantized once, not once per target."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    target = _target_tree(fused=True)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    with mock.patch.object(
        weight_converter, "quantize_weight_per_channel", wraps=weight_converter.quantize_weight_per_channel
    ) as quantize:
      converter.convert(_source_tree(True), target_state=target)

    fp8_groups = [g for g in converter._direct._groups if g.quantize_to_fp8]  # pylint: disable=protected-access
    num_targets = sum(len(g.targets) for g in fp8_groups)
    num_blocks = sum(len({idx for idx, _ in g.targets}) for g in fp8_groups)
    self.assertEqual(num_targets, 2 * num_blocks)  # a weight and a scale per block
    self.assertEqual(quantize.call_count, num_blocks)

  def test_case_15_gpt_oss_path_fuses_and_gets_fp8_scales(self):
    """GptOssMlp has no 'moe' or 'expert' in its name, but rank 4 (unrolled rank 3) identifies it as MoE."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    for slot in range(CYCLE):
      block = source["base"]["decoder"]["layers"][f"layer_{slot}"]
      block["GptOssMlp"] = block.pop("moe_block")

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out

    for layer in range(NUM_LAYERS):
      gpt_mlp = out_root["decoder"][f"layers_{layer}"]["GptOssMlp"]
      self.assertIn("wi", gpt_mlp)
      self.assertIn("wi_scale", gpt_mlp)
      self.assertIn("wo", gpt_mlp)
      self.assertIn("wo_scale", gpt_mlp)
      self.assertEqual(getattr(gpt_mlp["wi"], "value", gpt_mlp["wi"]).dtype, jnp.float8_e4m3fn)
      self.assertEqual(getattr(gpt_mlp["wi_scale"], "value", gpt_mlp["wi_scale"]).dtype, jnp.float32)

  def test_case_16_dense_mlp_and_shared_experts_get_no_fp8_scales(self):
    """Dense mlp and shared_experts have unrolled rank 2, so fp8_moe=True does not quantize or add scales."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    for slot in range(CYCLE):
      block = source["base"]["decoder"]["layers"][f"layer_{slot}"]
      block["mlp"] = {
          "wi_0": _scanned(EMB, MOE_DIM, offset=1000),
          "wi_1": _scanned(EMB, MOE_DIM, offset=2000),
          "wo": _scanned(MOE_DIM, EMB, offset=3000),
      }
      block["shared_experts"] = {
          "wi_0": _scanned(EMB, MOE_DIM, offset=4000),
          "wi_1": _scanned(EMB, MOE_DIM, offset=5000),
          "wo": _scanned(MOE_DIM, EMB, offset=6000),
      }

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out

    for layer in range(NUM_LAYERS):
      for mod in ("mlp", "shared_experts"):
        sub = out_root["decoder"][f"layers_{layer}"][mod]
        self.assertNotIn("wi_scale", sub)
        self.assertNotIn("wo_scale", sub)
        self.assertNotIn("wi_0_scale", sub)
        self.assertNotIn("wi_1_scale", sub)
        for key in sub:
          self.assertNotEqual(getattr(sub[key], "value", sub[key]).dtype, jnp.float8_e4m3fn)
          self.assertEqual(getattr(sub[key], "value", sub[key]).dtype, jnp.float32)

  def test_case_17_nested_scan_target_free_fp8(self):
    """Verifies target-free FP8 MoE conversion on the production nested scan layout."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    nested_source = _nested_source_tree(True)
    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    out = converter.convert(nested_source, target_state=None)
    out_root = out["base"] if "base" in out else out

    converter_unquantized = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=False)
    ref_out = converter_unquantized.convert(nested_source, target_state=None)
    ref_root = ref_out["base"] if "base" in ref_out else ref_out

    for layer in range(NUM_LAYERS):
      moe = out_root["decoder"][f"layers_{layer}"]["moe_block"]
      ref_moe = ref_root["decoder"][f"layers_{layer}"]["moe_block"]
      self.assertIn("wi", moe)
      self.assertIn("wi_scale", moe)
      self.assertIn("wo", moe)
      self.assertIn("wo_scale", moe)

      wi = getattr(moe["wi"], "value", moe["wi"])
      wi_scale = getattr(moe["wi_scale"], "value", moe["wi_scale"])
      wo = getattr(moe["wo"], "value", moe["wo"])
      wo_scale = getattr(moe["wo_scale"], "value", moe["wo_scale"])

      self.assertEqual(wi.dtype, jnp.float8_e4m3fn)
      self.assertEqual(wi.shape, (EXPERTS, EMB, MOE_DIM * 2))
      self.assertEqual(wi_scale.dtype, jnp.float32)
      self.assertEqual(wi_scale.shape, (EXPERTS, 1, MOE_DIM * 2))

      self.assertEqual(wo.dtype, jnp.float8_e4m3fn)
      self.assertEqual(wo.shape, (EXPERTS, MOE_DIM, EMB))
      self.assertEqual(wo_scale.dtype, jnp.float32)
      self.assertEqual(wo_scale.shape, (EXPERTS, 1, EMB))

      self.assertTrue(bool(jnp.all(wi_scale > 0)))
      self.assertTrue(bool(jnp.all(wo_scale > 0)))

      ref_wi = getattr(ref_moe["wi"], "value", ref_moe["wi"])
      ref_wo = getattr(ref_moe["wo"], "value", ref_moe["wo"])
      wi_dequant = jnp.asarray(wi, dtype=jnp.float32) * wi_scale
      wo_dequant = jnp.asarray(wo, dtype=jnp.float32) * wo_scale
      np.testing.assert_allclose(np.asarray(wi_dequant), np.asarray(ref_wi), rtol=0.07, atol=0.05)
      np.testing.assert_allclose(np.asarray(wo_dequant), np.asarray(ref_wo), rtol=0.07, atol=0.05)

  def test_case_18_target_dtype_none_preserves_exact_dtypes(self):
    """When target_dtype=None (as set by Tunix RL orchestrator), leaves preserve exact trainer dtypes."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    # Give gate float32 dtype and attention bfloat16 dtype
    source["base"]["decoder"]["layers"]["layer_0"]["moe_block"]["gate"]["kernel"] = jnp.ones(
        source["base"]["decoder"]["layers"]["layer_0"]["moe_block"]["gate"]["kernel"].shape,
        dtype=jnp.float32,
    )
    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", target_dtype=None)
    converter._direct.target_dtype = None  # pylint: disable=protected-access
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out

    gate_kernel = out_root["decoder"]["layers_0"]["moe_block"]["gate"]["kernel"]
    self.assertEqual(getattr(gate_kernel, "value", gate_kernel).dtype, jnp.float32)

  def test_case_19_moe_tp_size_1_uses_single_shard_even_when_kv_tp_1(self):
    """For an unsharded source, moe_mlp_tp_size=1 must produce n_shards=1 even when kv_tp_size=1 and tp=1."""
    cfg = _config(prefuse_moe_weights=True, kv_tp_size=1, moe_mlp_tp_size=1)
    source = _source_tree(True)
    src_moe = source["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    expected_wi_0 = np.asarray(jnp.take(src_moe["wi_0"], 0, axis=SCAN_AXIS))
    expected_wi_1 = np.asarray(jnp.take(src_moe["wi_1"], 0, axis=SCAN_AXIS))
    expected_concat = np.concatenate([expected_wi_0, expected_wi_1], axis=-1)

    converter = weight_converter.WeightConverter(config=cfg, tp=1, rollout_backend="maxtext")
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    wi_tf = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
        "value",
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
    )
    np.testing.assert_allclose(np.asarray(wi_tf), expected_concat)

  def test_case_20_default_tp_rollout_uses_tp_shards(self):
    """Under default TP rollout (tp=2, kv_tp_size=1, moe_mlp_tp_size=1), n_shards falls back to tp=2."""
    cfg = _config(prefuse_moe_weights=True, kv_tp_size=1, moe_mlp_tp_size=1)
    source = _source_tree(True)
    converter = weight_converter.WeightConverter(config=cfg, tp=2, rollout_backend="maxtext")
    out = converter.convert(source, target_state=None)
    out_root = out["base"] if "base" in out else out
    wi_tf = getattr(
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
        "value",
        out_root["decoder"]["layers_0"]["moe_block"]["wi"],
    )
    src_moe = source["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    expected_wi_0 = np.asarray(jnp.take(src_moe["wi_0"], 0, axis=SCAN_AXIS))
    expected_wi_1 = np.asarray(jnp.take(src_moe["wi_1"], 0, axis=SCAN_AXIS))
    plain_concat = np.concatenate([expected_wi_0, expected_wi_1], axis=-1)
    self.assertFalse(np.allclose(np.asarray(wi_tf), plain_concat))

  def test_case_21_fp8_moe_without_scale_target_raises(self):
    """Target-based conversion with fp8_moe=True into a rollout tree without scales raises ConversionPlanError."""
    cfg = _config(inhomogeneous_layer_cycle_interval=CYCLE, num_decoder_layers=NUM_LAYERS, prefuse_moe_weights=True)
    source = _source_tree(True)
    target = _target_tree(fused=True)  # target has no _scale leaves
    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    with self.assertRaisesRegex(weight_converter.ConversionPlanError, "has no corresponding FP8 scale"):
      converter.convert(source, target_state=target)

  def test_case_22_target_free_fp8_quantizes_from_bf16(self):
    """In target-free mode with float32 trainer weights and target_dtype=None, FP8 quantizes from bfloat16."""
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source_f32 = _source_tree(True)
    moe = source_f32["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    moe["wi_0"] = moe["wi_0"].astype(jnp.float32)
    moe["wi_1"] = moe["wi_1"].astype(jnp.float32)
    moe["wo"] = moe["wo"].astype(jnp.float32)

    source_bf16 = _source_tree(True)
    moe_bf16 = source_bf16["base"]["decoder"]["layers"]["layer_0"]["moe_block"]
    moe_bf16["wi_0"] = moe["wi_0"].astype(jnp.bfloat16)
    moe_bf16["wi_1"] = moe["wi_1"].astype(jnp.bfloat16)
    moe_bf16["wo"] = moe["wo"].astype(jnp.bfloat16)

    converter_f32 = weight_converter.WeightConverter(
        config=cfg, rollout_backend="maxtext", fp8_moe=True, target_dtype=None
    )
    converter_f32._direct.target_dtype = None  # pylint: disable=protected-access
    out_f32 = converter_f32.convert(source_f32, target_state=None)

    converter_bf16 = weight_converter.WeightConverter(
        config=cfg, rollout_backend="maxtext", fp8_moe=True, target_dtype=None
    )
    converter_bf16._direct.target_dtype = None  # pylint: disable=protected-access
    out_bf16 = converter_bf16.convert(source_bf16, target_state=None)

    out_root_f32 = out_f32["base"] if "base" in out_f32 else out_f32
    out_root_bf16 = out_bf16["base"] if "base" in out_bf16 else out_bf16
    wi_f32 = getattr(
        out_root_f32["decoder"]["layers_0"]["moe_block"]["wi"],
        "value",
        out_root_f32["decoder"]["layers_0"]["moe_block"]["wi"],
    )
    wi_bf16 = getattr(
        out_root_bf16["decoder"]["layers_0"]["moe_block"]["wi"],
        "value",
        out_root_bf16["decoder"]["layers_0"]["moe_block"]["wi"],
    )
    wi_scale_f32 = getattr(
        out_root_f32["decoder"]["layers_0"]["moe_block"]["wi_scale"],
        "value",
        out_root_f32["decoder"]["layers_0"]["moe_block"]["wi_scale"],
    )
    wi_scale_bf16 = getattr(
        out_root_bf16["decoder"]["layers_0"]["moe_block"]["wi_scale"],
        "value",
        out_root_bf16["decoder"]["layers_0"]["moe_block"]["wi_scale"],
    )
    np.testing.assert_array_equal(np.asarray(wi_f32), np.asarray(wi_bf16))
    np.testing.assert_array_equal(np.asarray(wi_scale_f32), np.asarray(wi_scale_bf16))

  def test_case_23_fp8_moe_without_src_flat_raises(self):
    """Grouping an FP8 plan without src_flat fails loudly rather than misreporting fp8_moe."""
    cfg = _config(inhomogeneous_layer_cycle_interval=CYCLE, num_decoder_layers=NUM_LAYERS, prefuse_moe_weights=True)
    target = _target_tree(fused=True)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    converter.convert(_source_tree(True), target_state=target)
    direct = converter._direct  # pylint: disable=protected-access
    with self.assertRaisesRegex(weight_converter.ConversionPlanError, "without src_flat"):
      _group_plan(direct._plan, fp8_moe=True)  # pylint: disable=protected-access

  def test_case_24_fp8_scale_on_non_expert_target_raises(self):
    """Rollout target having a scale leaf on a non-expert tensor raises ConversionPlanError."""
    cfg = _config(inhomogeneous_layer_cycle_interval=CYCLE, num_decoder_layers=NUM_LAYERS, prefuse_moe_weights=True)
    target = _target_tree(fused=True)
    for layer in range(NUM_LAYERS):
      moe = target["decoder"][f"layers_{layer}"]["moe_block"]
      moe["wi"] = moe["wi"].astype(jnp.float8_e4m3fn)
      moe["wi_scale"] = jnp.ones((EXPERTS, 1, MOE_DIM * 2), dtype=jnp.float32)
      moe["wo"] = moe["wo"].astype(jnp.float8_e4m3fn)
      moe["wo_scale"] = jnp.ones((EXPERTS, 1, EMB), dtype=jnp.float32)
      # Non-expert 2D gate kernel unexpectedly having a scale in target
      moe["gate"]["kernel_scale"] = jnp.ones((1, EXPERTS), dtype=jnp.float32)

    converter = weight_converter.WeightConverter(config=cfg, rollout_backend="maxtext", fp8_moe=True)
    with self.assertRaisesRegex(weight_converter.ConversionPlanError, "not an FP8-quantized MoE expert weight"):
      converter.convert(_source_tree(True), target_state=target)


def _vllm_torch_hybrid_source_and_layout(
    *,
    num_layers: int = 6,
    cycle: int = 4,
    moe: bool = True,
    prefused_wi: bool = False,
    tp_size: int = 2,
    use_ep: bool = False,
    padded_moe_interm_per_rank: int = MOE_DIM // 2,
    qkvz_flipped: bool = False,
    conv_transposed: bool = False,
    logits_via_embedding: bool = False,
    emit_lm_head: bool = True,
):
  """Constructs a real-naming Qwen3.5 trainer tree + rank-0 `param_layouts`."""
  hd = 4
  nq, nkv = 4, 2
  hk, hv, dk, dv, conv_k = 2, 4, 4, 4, 4
  num_full_blocks = num_layers // cycle
  rem_layers = num_layers % cycle

  def _make_gdn_sub(scan_dims: tuple[int, ...], seed: int):
    def _s(*dims, off=0):
      shape = list(dims)
      for s_len in reversed(scan_dims):
        shape.insert(SCAN_AXIS, s_len)
      return _arr(*shape, offset=seed + off)

    d = {
        "input_layernorm": {"scale": _s(EMB, off=10)},
        "post_attention_layernorm": {"scale": _s(EMB, off=20)},
        "attention": {
            "in_proj_qkvz": {"kernel": _s(EMB, 2 * hk * dk + 2 * hv * dv, off=30)},
            "in_proj_ba": {"kernel": _s(EMB, 2 * hv, off=40)},
            "conv1d": {"kernel": _s(conv_k, 1, 2 * hk * dk + hv * dv, off=50)},
            "dt_bias": _s(hv, off=60),
            "A_log": _s(hv, off=70),
            "norm": {"scale": _s(dv, off=80)},
            "out_proj": {"kernel": _s(hv * dv, EMB, off=90)},
        },
    }
    if moe:
      moe_d = {
          "gate": {"kernel": _s(EMB, EXPERTS, off=100)},
          "wo": _s(EXPERTS, MOE_DIM, EMB, off=130),
          "shared_experts": {
              "wi_0": {"kernel": _s(EMB, MOE_DIM, off=140)},
              "wi_1": {"kernel": _s(EMB, MOE_DIM, off=150)},
              "wo": {"kernel": _s(MOE_DIM, EMB, off=160)},
          },
          "shared_expert_gate": {"kernel": _s(EMB, 1, off=170)},
      }
      if prefused_wi:
        w0 = _s(EXPERTS, EMB, MOE_DIM, off=110)
        w1 = _s(EXPERTS, EMB, MOE_DIM, off=120)
        moe_d["wi"] = jnp.concatenate([w0, w1], axis=-1)
      else:
        moe_d["wi_0"] = _s(EXPERTS, EMB, MOE_DIM, off=110)
        moe_d["wi_1"] = _s(EXPERTS, EMB, MOE_DIM, off=120)
      d["moe_block"] = moe_d
    else:
      d["mlp"] = {
          "wi_0": {"kernel": _s(EMB, MOE_DIM, off=110)},
          "wi_1": {"kernel": _s(EMB, MOE_DIM, off=120)},
          "wo": {"kernel": _s(MOE_DIM, EMB, off=130)},
      }
    return d

  def _make_full_sub(scan_dims: tuple[int, ...], seed: int):
    def _s(*dims, off=0):
      shape = list(dims)
      for s_len in reversed(scan_dims):
        shape.insert(SCAN_AXIS, s_len)
      return _arr(*shape, offset=seed + off)

    d = {
        "input_layernorm": {"scale": _s(EMB, off=10)},
        "post_attention_layernorm": {"scale": _s(EMB, off=20)},
        "attention": {
            "attention": {
                "query": {"kernel": _s(EMB, nq, 2 * hd, off=30)},
                "key": {"kernel": _s(EMB, nkv, hd, off=40)},
                "value": {"kernel": _s(EMB, nkv, hd, off=50)},
                "query_norm": {"scale": _s(hd, off=60)},
                "key_norm": {"scale": _s(hd, off=70)},
                "out": {"kernel": _s(nq * hd, EMB, off=80)},
            }
        },
    }
    if moe:
      d["moe_block"] = _make_gdn_sub(scan_dims, seed)["moe_block"]
    else:
      d["mlp"] = _make_gdn_sub(scan_dims, seed)["mlp"]
    return d

  decoder: dict = {
      "decoder_norm": {"scale": _arr(EMB, offset=5)},
  }
  if num_full_blocks > 0:
    decoder["layers"] = {
        "local_layers": _make_gdn_sub((num_full_blocks, cycle - 1), seed=1000),
        "global_layer": _make_full_sub((num_full_blocks,), seed=2000),
    }
  if rem_layers > 0:
    decoder["layers_remainder"] = {
        "local_layers": _make_gdn_sub((rem_layers,), seed=3000),
    }
  if not logits_via_embedding:
    decoder["logits_dense"] = {"kernel": _arr(EMB, 16, offset=9)}

  source = {
      "base": {
          "token_embedder": {"embedding": _arr(16, EMB, offset=1)},
          "decoder": decoder,
      }
  }

  def _entry(local_shape, shard_dim=None, num_shards=1, domain="replicated", **extra):
    global_shape = list(local_shape)
    if shard_dim is not None and num_shards > 1:
      global_shape[shard_dim] *= num_shards
    spec = {
        "local_shape": list(local_shape),
        "global_shape": global_shape,
        "dtype": "float32",
        "shard_dim": shard_dim,
        "num_shards": num_shards,
        "shard_domain": domain,
        "flipped": False,
        "conv_transposed": False,
    }
    spec.update(extra)
    return spec

  layouts: dict = {
      "embed_tokens.weight": _entry([16 // tp_size, EMB], 0, tp_size, "tp"),
      "norm.weight": _entry([EMB]),
  }
  if emit_lm_head:
    layouts["lm_head.weight"] = _entry([16 // tp_size, EMB], 0, tp_size, "tp")

  k_local = max(1, hk // tp_size)
  v_local = hv // tp_size
  kq_rep = max(1, tp_size // hk)
  kv_rep = max(1, tp_size // nkv)
  nkv_local = max(1, nkv // tp_size)
  nq_local = nq // tp_size

  for lyr in range(num_layers):
    pfx = f"layers.{lyr}."
    layouts[pfx + "input_layernorm.weight"] = _entry([EMB])
    layouts[pfx + "post_attention_layernorm.weight"] = _entry([EMB])
    if lyr % cycle == cycle - 1:
      layouts[pfx + "self_attn.qkv_proj.weight"] = _entry(
          [(nq_local * 2 + 2 * nkv_local) * hd, EMB],
          0,
          tp_size,
          "tp",
          checkpoint_sizes=[nq * 2 * hd, nkv * hd, nkv * hd],
          replication_factors=[1, kv_rep, kv_rep],
      )
      layouts[pfx + "self_attn.q_norm.weight"] = _entry([hd])
      layouts[pfx + "self_attn.k_norm.weight"] = _entry([hd])
      layouts[pfx + "self_attn.o_proj.weight"] = _entry([EMB, nq_local * hd], 1, tp_size, "tp")
    else:
      qkvz_out_local = 2 * k_local * dk + 2 * v_local * dv
      qkvz_shape = [EMB, qkvz_out_local] if qkvz_flipped else [qkvz_out_local, EMB]
      layouts[pfx + "linear_attn.in_proj_qkvz.weight"] = _entry(
          qkvz_shape,
          1 if qkvz_flipped else 0,
          tp_size,
          "tp",
          flipped=qkvz_flipped,
          checkpoint_sizes=[hk * dk, hk * dk, hv * dv, hv * dv],
          replication_factors=[kq_rep, kq_rep, 1, 1],
      )
      layouts[pfx + "linear_attn.in_proj_ba.weight"] = _entry(
          [2 * v_local, EMB],
          0,
          tp_size,
          "tp",
          checkpoint_sizes=[hv, hv],
          replication_factors=[1, 1],
      )
      c_local = 2 * k_local * dk + v_local * dv
      conv_shape = [conv_k, 1, c_local] if conv_transposed else [c_local, 1, conv_k]
      layouts[pfx + "linear_attn.conv1d.weight"] = _entry(
          conv_shape,
          2 if conv_transposed else 0,
          tp_size,
          "tp",
          conv_transposed=conv_transposed,
      )
      layouts[pfx + "linear_attn.dt_bias"] = _entry([v_local], 0, tp_size, "tp")
      layouts[pfx + "linear_attn.A_log"] = _entry([v_local], 0, tp_size, "tp")
      layouts[pfx + "linear_attn.norm.weight"] = _entry([dv])
      layouts[pfx + "linear_attn.out_proj.weight"] = _entry([EMB, v_local * dv], 1, tp_size, "tp")

    if moe:
      layouts[pfx + "mlp.gate.weight"] = _entry([EXPERTS, EMB])
      if use_ep:
        layouts[pfx + "mlp.experts.w13_weight"] = _entry([EXPERTS // tp_size, 2 * MOE_DIM, EMB], 0, tp_size, "ep")
        layouts[pfx + "mlp.experts.w2_weight"] = _entry([EXPERTS // tp_size, EMB, MOE_DIM], 0, tp_size, "ep")
      else:
        layouts[pfx + "mlp.experts.w13_weight"] = _entry([EXPERTS, 2 * padded_moe_interm_per_rank, EMB], 1, tp_size, "tp")
        layouts[pfx + "mlp.experts.w2_weight"] = _entry([EXPERTS, EMB, padded_moe_interm_per_rank], 2, tp_size, "tp")
      layouts[pfx + "mlp.shared_expert.gate_up_proj.weight"] = _entry([2 * (MOE_DIM // tp_size), EMB], 0, tp_size, "tp")
      layouts[pfx + "mlp.shared_expert.down_proj.weight"] = _entry([EMB, MOE_DIM // tp_size], 1, tp_size, "tp")
      layouts[pfx + "mlp.shared_expert_gate.weight"] = _entry([1, EMB])
    else:
      layouts[pfx + "mlp.gate_up_proj.weight"] = _entry([2 * (MOE_DIM // tp_size), EMB], 0, tp_size, "tp")
      layouts[pfx + "mlp.down_proj.weight"] = _entry([EMB, MOE_DIM // tp_size], 1, tp_size, "tp")
  return source, layouts


class VllmTorchWeightConverterTest(unittest.TestCase):
  """Unit tests for `rollout_backend='vllm_torch'` layout-driven conversion."""

  def _cfg(self, num_layers=6, cycle=4, decoder_block="qwen3_5"):
    return _config(
        decoder_block=decoder_block,
        num_decoder_layers=num_layers,
        inhomogeneous_layer_cycle_interval=cycle,
        scan_layers=True,
        emb_dim=EMB,
        head_dim=4,
        num_query_heads=4,
        num_kv_heads=2,
        base_moe_mlp_dim=MOE_DIM,
        moe_mlp_dim=MOE_DIM,
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        gdn_key_head_dim=4,
        gdn_value_head_dim=4,
        gdn_conv_kernel_dim=4,
        dtype="float32",
        weight_dtype="float32",
        rollout_weight_dtype="",
    )

  @staticmethod
  def _flat_arrays(tree):
    return {
        ".".join(str(p) for p in k): v.get_value() if hasattr(v, "get_value") else v
        for k, v in traverse_util.flatten_dict(tree).items()
    }

  def test_default_ep_and_tp_moe_synthesis_and_target_state_forwarding(self):
    source, _ = _vllm_torch_hybrid_source_and_layout(
        num_layers=6,
        cycle=4,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    cfg = self._cfg(num_layers=6, cycle=4)
    cfg.rollout_weight_dtype = "bfloat16"
    converter_ep = WeightConverter(
        config=cfg,
        tp=2,
        rollout_backend="vllm_torch",
    )
    flat_ep = self._flat_arrays(converter_ep.convert(source))
    self.assertEqual(flat_ep["embed_tokens.weight"].shape, (16, EMB))
    self.assertEqual(flat_ep["embed_tokens.weight"].dtype, jnp.bfloat16)
    self.assertEqual(flat_ep["layers.0.linear_attn.in_proj_qkvz.weight"].shape, (48, EMB))
    self.assertEqual(flat_ep["layers.0.linear_attn.in_proj_ba.weight"].shape, (EMB, 8))
    self.assertEqual(flat_ep["layers.0.linear_attn.A_log"].dtype, jnp.float32)
    self.assertEqual(flat_ep["layers.3.self_attn.qkv_proj.weight"].shape, (EMB, 48))
    self.assertEqual(flat_ep["layers.0.mlp.experts.w13_weight"].shape, (EXPERTS, EMB, 256))
    self.assertEqual(flat_ep["layers.0.mlp.experts.w2_weight"].shape, (EXPERTS, 128, EMB))

    # `target_state` dict fallback and synthesized TP-MoE (`moe_mlp_tp_size == tp`).
    src_tp, tp_layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        tp_size=2,
        use_ep=False,
        padded_moe_interm_per_rank=128,
        qkvz_flipped=True,
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    via_kwarg = self._flat_arrays(conv.convert(copy.deepcopy(src_tp), target_layout=tp_layouts))
    via_state = self._flat_arrays(conv.convert(copy.deepcopy(src_tp), target_state=tp_layouts))
    for k, arr in via_kwarg.items():
      np.testing.assert_array_equal(np.asarray(arr), np.asarray(via_state[k]))

    conv_tp = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        moe_mlp_tp_size=2,
        rollout_backend="vllm_torch",
    )
    flat_tp = self._flat_arrays(conv_tp.convert(copy.deepcopy(src_tp)))
    self.assertEqual(flat_tp["layers.0.mlp.experts.w13_weight"].shape, (EXPERTS, EMB, 4 * 128))
    self.assertEqual(flat_tp["layers.0.mlp.experts.w2_weight"].shape, (EXPERTS, 2 * 128, EMB))

  def test_nested_scan_remainder_only_and_flipped_kernels(self):
    source, layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=6,
        cycle=4,
        moe=True,
        tp_size=2,
        use_ep=False,
        padded_moe_interm_per_rank=8,
        qkvz_flipped=True,
        conv_transposed=True,
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=6, cycle=4),
        tp=2,
        rollout_backend="vllm_torch",
    )
    flat_out = self._flat_arrays(conv.convert(source, target_layout=layouts))
    self.assertEqual(set(flat_out), set(layouts))
    rem_scale = source["base"]["decoder"]["layers_remainder"]["local_layers"]["input_layernorm"]["scale"]
    np.testing.assert_array_equal(
        np.asarray(flat_out["layers.5.input_layernorm.weight"]),
        np.asarray(jnp.take(rem_scale, 1, axis=SCAN_AXIS)),
    )
    w13 = np.asarray(flat_out["layers.0.mlp.experts.w13_weight"])
    w2 = np.asarray(flat_out["layers.0.mlp.experts.w2_weight"])
    np.testing.assert_array_equal(w13[:, 3:8, :], 0.0)
    np.testing.assert_array_equal(w2[:, :, 3:8], 0.0)

    # Remainder-only hybrid scan when `num_decoder_layers < cycle` (`layers` absent).
    src_rem, _ = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=4,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    self.assertNotIn("layers", src_rem["base"]["decoder"])
    conv_rem = WeightConverter(
        config=self._cfg(num_layers=2, cycle=4),
        tp=2,
        rollout_backend="vllm_torch",
    )
    rem_only_out = self._flat_arrays(conv_rem.convert(src_rem))
    self.assertIn("layers.1.linear_attn.in_proj_qkvz.weight", rem_only_out)

  def test_moe_dim_scaling_prefused_wi_and_multimodal_skip(self):
    src_split, layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        prefused_wi=False,
        tp_size=2,
        use_ep=True,
    )
    src_fused, _ = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        prefused_wi=True,
        tp_size=2,
        use_ep=True,
    )
    src_fused["base"]["audio_encoder"] = {
        "conv2d1": {"kernel": jnp.ones((3, 3, 1, 8), dtype=jnp.float32)},
    }
    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    out_split = self._flat_arrays(conv.convert(copy.deepcopy(src_split), target_layout=layouts))
    out_fused = self._flat_arrays(conv.convert(src_fused, target_layout=layouts))
    self.assertTrue(all("audio_encoder" not in k for k in out_fused))
    for k, arr in out_split.items():
      np.testing.assert_array_equal(np.asarray(arr), np.asarray(out_fused[k]))

    # Scaled `moe_mlp_dim` wins over `base_moe_mlp_dim` (including `-1` sentinel).
    ref_w13 = self._flat_arrays(conv.convert(copy.deepcopy(src_split)))["layers.0.mlp.experts.w13_weight"]
    for base_dim in (-1, MOE_DIM // 2):
      cfg = self._cfg(num_layers=2, cycle=2)
      cfg.base_moe_mlp_dim = base_dim
      cfg.moe_mlp_dim = MOE_DIM
      scaled_conv = WeightConverter(
          config=cfg,
          tp=2,
          rollout_backend="vllm_torch",
      )
      got = self._flat_arrays(scaled_conv.convert(copy.deepcopy(src_split)))
      np.testing.assert_array_equal(np.asarray(got["layers.0.mlp.experts.w13_weight"]), np.asarray(ref_w13))

  def test_dense_model_and_tied_embedding_deduplication(self):
    src_dedup, layouts_dedup = _vllm_torch_hybrid_source_and_layout(
        num_layers=4,
        cycle=4,
        moe=False,
        tp_size=2,
        logits_via_embedding=True,
        emit_lm_head=False,
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=4, cycle=4),
        tp=2,
        rollout_backend="vllm_torch",
    )
    flat_names = set(self._flat_arrays(conv.convert(src_dedup, target_layout=layouts_dedup)))
    self.assertNotIn("lm_head.weight", flat_names)
    self.assertIn("layers.0.mlp.gate_up_proj.weight", flat_names)
    self.assertIn("layers.0.mlp.down_proj.weight", flat_names)

  def test_multi_axis_out_sharding_on_8_cpu_mesh(self):
    from maxtext.integration.vllm.vllm_torch_converter import _multi_axis_out_sharding  # pylint: disable=import-outside-toplevel

    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)
    mesh = jax.sharding.Mesh(np.array(devices[:8]).reshape((2, 2, 2)), ("expert", "fsdp", "tensor"))
    mesh_2d = jax.sharding.Mesh(np.array(devices[:8]).reshape((4, 2)), ("expert", "fsdp"))
    self.assertEqual(
        _multi_axis_out_sharding(mesh_2d, (256, 2048, 1024), param_name="mlp.experts.w13_weight").spec,
        jax.sharding.PartitionSpec("expert", "fsdp", None),
    )
    self.assertEqual(
        _multi_axis_out_sharding(mesh_2d, (256, 512, 2048), param_name="mlp.experts.w2_weight").spec,
        jax.sharding.PartitionSpec("expert", None, "fsdp"),
    )
    # Non-expert tensors prefer `fsdp` ahead of `expert`.
    self.assertEqual(
        _multi_axis_out_sharding(mesh_2d, (2048,), param_name="norm.weight").spec,
        jax.sharding.PartitionSpec("fsdp"),
    )

    src, layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    sharded_src = jax.tree.map(
        lambda x: jax.device_put(
            x,
            jax.sharding.NamedSharding(
                mesh,
                jax.sharding.PartitionSpec("fsdp" if x.shape[0] % 2 == 0 else None),
            ),
        ),
        src,
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    ref_out = self._flat_arrays(conv.convert(src, target_layout=layouts))
    shd_out = self._flat_arrays(conv.convert(sharded_src, target_layout=layouts))
    for k, ref_arr in ref_out.items():
      np.testing.assert_allclose(np.asarray(shd_out[k]), np.asarray(ref_arr))
      if k.endswith(("experts.w13_weight", "experts.w2_weight")):
        self.assertEqual(shd_out[k].sharding.spec[0], "expert")

  def test_tristate_flipped_cache_and_replication_override(self):
    src, base_layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    layouts_omitted = copy.deepcopy(base_layouts)
    layouts_omitted["layers.1.self_attn.qkv_proj.weight"].update(
        {
            "local_shape": [24, EMB],
            "global_shape": [96, EMB],
            "num_shards": 4,
            "replication_factors": [1, 4, 4],
        }
    )
    for lyr in range(2):
      layouts_omitted[f"layers.{lyr}.mlp.experts.w2_weight"] = {
          "local_shape": [EXPERTS // 2, EMB, EMB],
          "global_shape": [EXPERTS, EMB, EMB],
          "dtype": "float32",
          "shard_dim": 0,
          "num_shards": 2,
          "shard_domain": "ep",
      }
    layouts_false = copy.deepcopy(layouts_omitted)
    layouts_false["layers.0.mlp.experts.w2_weight"]["flipped"] = False

    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    out_omitted = self._flat_arrays(conv.convert(copy.deepcopy(src), target_layout=layouts_omitted))
    out_false = self._flat_arrays(conv.convert(copy.deepcopy(src), target_layout=layouts_false))
    np.testing.assert_array_equal(
        np.swapaxes(np.asarray(out_omitted["layers.0.mlp.experts.w2_weight"]), 1, 2),
        np.asarray(out_false["layers.0.mlp.experts.w2_weight"]),
    )
    qkv_arr = np.asarray(out_omitted["layers.1.self_attn.qkv_proj.weight"]).reshape(4, 24, EMB)
    np.testing.assert_array_equal(qkv_arr[0, 8:, :], qkv_arr[3, 8:, :])

  def test_nnx_state_and_non_param_cache_variables(self):
    src, layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    src["base"]["decoder"]["layers"]["local_layers"]["rng_step"] = nnx.Cache(jnp.zeros((1, 1), dtype=jnp.int32))
    nnx_state = nnx.State(
        jax.tree.map(
            lambda x: x if isinstance(x, nnx.Variable) else nnx.Param(x),
            src,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    self.assertIsNotNone(conv.convert(nnx_state, target_layout=layouts))

  def test_fail_closed_config_layout_and_layer_idx_validation(self):
    src, base_layouts = _vllm_torch_hybrid_source_and_layout(
        num_layers=2,
        cycle=2,
        moe=True,
        tp_size=2,
        use_ep=True,
    )
    conv = WeightConverter(
        config=self._cfg(num_layers=2, cycle=2),
        tp=2,
        rollout_backend="vllm_torch",
    )
    for bad_block in ("llama2", "qwen3_moe"):
      with self.assertRaisesRegex(ValueError, "decoder_block"):
        WeightConverter(
            config=self._cfg(num_layers=2, cycle=2, decoder_block=bad_block),
            tp=2,
            rollout_backend="vllm_torch",
        )
    with self.assertRaisesRegex(ValueError, "fp8_moe"):
      WeightConverter(
          config=self._cfg(num_layers=2, cycle=2),
          tp=2,
          rollout_backend="vllm_torch",
          fp8_moe=True,
      )
    with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
      WeightConverter(
          config=self._cfg(num_layers=2, cycle=2),
          tp=2,
          rollout_backend="vllm_torch",
          target_dtype="float8_e4m3fn",
      )
    for bad_flag in ("moe_flat_fsdp_weights", "moe_use_fused_gmm3d", "fused_qkv"):
      with self.subTest(bad_flag=bad_flag):
        cfg_flag = self._cfg(num_layers=2, cycle=2)
        setattr(cfg_flag, bad_flag, True)
        with self.assertRaisesRegex(ValueError, bad_flag):
          WeightConverter(config=cfg_flag, tp=2, rollout_backend="vllm_torch")

    # Query head count (not just q_total) must be divisible by rollout tp.
    src_q3 = copy.deepcopy(src)
    q_key = src_q3["base"]["decoder"]["layers"]["global_layer"]["attention"]["attention"]["query"]
    q_key["kernel"] = jnp.ones((1, EMB, 3, 8), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "num_q_heads=3"):
      conv.convert(src_q3)

    # Out-of-range `moe_mlp_dim` must fail closed instead of silently clamping.
    cfg_big_moe = self._cfg(num_layers=2, cycle=2)
    cfg_big_moe.moe_mlp_dim = MOE_DIM + 4
    conv_big_moe = WeightConverter(config=cfg_big_moe, tp=2, rollout_backend="vllm_torch")
    with self.assertRaisesRegex(ValueError, "exceeds"):
      conv_big_moe.convert(copy.deepcopy(src), target_layout=base_layouts)

    # Passing a nested state pytree as `target_state` raises a descriptive ValueError.
    with self.assertRaisesRegex(ValueError, "Invalid target_layout entry"):
      conv.convert(copy.deepcopy(src), target_state={"layers": {"0": {}}})

    for kwargs, err_re in (
        ({"dp": 3}, "num_experts=4"),
        ({"kv_tp_size": 4}, "kv_tp_size"),
        ({"moe_mlp_tp_size": 4}, "moe_mlp_tp_size"),
    ):
      with self.subTest(kwargs=kwargs):
        bad_conv = WeightConverter(
            config=self._cfg(num_layers=2, cycle=2),
            tp=2,
            rollout_backend="vllm_torch",
            **kwargs,
        )
        with self.assertRaisesRegex(ValueError, err_re):
          bad_conv.convert(copy.deepcopy(src))

    # Swapped `layer_idx` is caught against JAX pytree traversal order.
    sorted_keys = sorted(base_layouts, key=lambda k: tuple(k.split(".")))
    bad_idx = {k: {**base_layouts[k], "layer_idx": i} for i, k in enumerate(sorted_keys)}
    bad_idx[sorted_keys[0]]["layer_idx"] = 1
    bad_idx[sorted_keys[1]]["layer_idx"] = 0
    with self.assertRaisesRegex(ValueError, "layer_idx"):
      conv.convert(copy.deepcopy(src), target_layout=bad_idx)

    for key, patch_dict in (
        ("layers.1.self_attn.qkv_proj.weight", {"num_shards": 1, "replication_factors": [1, 2, 2]}),
        ("layers.1.self_attn.qkv_proj.weight", {"shard_dim": None, "num_shards": 4}),
        ("layers.0.linear_attn.conv1d.weight", {"num_shards": 0}),
        ("layers.0.mlp.experts.w13_weight", {"dtype": "torch.float8_e4m3fn"}),
        ("layers.0.mlp.experts.w13_weight", {"num_shards": 0}),
        (
            "layers.0.mlp.experts.w13_weight",
            {"shard_domain": "tp", "shard_dim": 2, "num_shards": 3, "local_shape": [EXPERTS, EMB, 8]},
        ),
        ("layers.0.mlp.experts.w13_weight", {"local_shape": [EXPERTS // 2, EMB, 25]}),
        ("layers.0.mlp.experts.w13_weight", {"local_shape": [EXPERTS // 2, EMB, 2 * (MOE_DIM - 2)]}),
        ("layers.0.mlp.experts.w2_weight", {"local_shape": [EXPERTS // 2, MOE_DIM - 2, EMB]}),
    ):
      bad_layouts = copy.deepcopy(base_layouts)
      bad_layouts[key] = {**bad_layouts[key], **patch_dict}
      with self.subTest(key=key, patch_dict=patch_dict):
        with self.assertRaises(ValueError):
          conv.convert(copy.deepcopy(src), target_layout=bad_layouts)


if __name__ == "__main__":
  unittest.main()

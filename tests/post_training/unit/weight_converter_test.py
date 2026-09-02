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

import os

# Must precede the first JAX import: the cross-mesh tests below need more than
# one CPU device, and the backend reads this only at initialization.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import types as pytypes  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

import logging
from typing import Any
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position
from flax import traverse_util  # pylint: disable=wrong-import-position

from maxtext.integration.vllm.weight_converter import (
    ConversionPlanError,
    MaxTextToMaxTextConverter,
    MoEFusedLayout,
    Rule,
    WeightConverter,
    MODEL_TO_CONVERSION_RULES,
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
  cfg = pytypes.SimpleNamespace(
      num_decoder_layers=NUM_LAYERS,
      inhomogeneous_layer_cycle_interval=CYCLE,
      param_scan_axis=SCAN_AXIS,
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


def _profile_conversion_worker(is_streaming: bool, result_queue: Any):
  import gc
  import resource
  import types as pytypes
  import jax.numpy as jnp
  from maxtext.integration.vllm.weight_converter import MaxTextToMaxTextConverter

  num_layers = 16
  cycle = 2
  scaled_emb = 128
  scaled_experts = 8
  scaled_mlp = 256
  blocks = num_layers // cycle

  cfg = pytypes.SimpleNamespace(
      inhomogeneous_layer_cycle_interval=cycle,
      num_decoder_layers=num_layers,
      param_scan_axis=1,
      padded_base_moe_mlp_dim=scaled_mlp,
      prefuse_moe_weights=True,
      weight_dtype=jnp.float32,
  )

  def _arr(*shape):
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
    from maxtext.integration.tunix.weight_mapping import raiden_unscan

    source = _source_tree(True)
    with self.assertRaises(ValueError) as ctx:
      raiden_unscan.unscan_layers(source, num_layers=NUM_LAYERS, scan_axis=SCAN_AXIS)
    self.assertIn("expected axis 1 to be num_layers=8", str(ctx.exception))

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
    self.assertIn("token_embedder", out)
    self.assertIn("decoder", out)
    for i in range(4):
      layer_key = f"layers_{i}"
      self.assertIn(layer_key, out["decoder"])
      scale = getattr(out["decoder"][layer_key]["input_layernorm"]["scale"], "value", out["decoder"][layer_key]["input_layernorm"]["scale"])
      query = getattr(out["decoder"][layer_key]["self_attention"]["query"]["kernel"], "value", out["decoder"][layer_key]["self_attention"]["query"]["kernel"])
      self.assertEqual(scale.shape, (EMB,))
      self.assertEqual(query.shape, (EMB, 2, 4))

  def test_case_2_hybrid_cycle_target_free_unroll(self):
    cfg = _config()
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    out = converter.convert(source, target_state=None)
    src_layers = source["base"]["decoder"]["layers"]
    for layer in range(NUM_LAYERS):
      slot, block = layer % CYCLE, layer // CYCLE
      got = getattr(out["decoder"][f"layers_{layer}"]["input_layernorm"]["scale"], "value", out["decoder"][f"layers_{layer}"]["input_layernorm"]["scale"])
      want = jnp.take(src_layers[f"layer_{slot}"]["input_layernorm"]["scale"], block, axis=SCAN_AXIS)
      np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  def test_case_3_prefused_moe_target_free(self):
    from maxtext.integration.vllm.moe_padding import compute_padded_moe_mlp_dim

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
    wi = getattr(out["decoder"]["layers_0"]["moe_block"]["wi"], "value", out["decoder"]["layers_0"]["moe_block"]["wi"])
    wo = getattr(out["decoder"]["layers_0"]["moe_block"]["wo"], "value", out["decoder"]["layers_0"]["moe_block"]["wo"])
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
    for leaf in jax.tree_util.tree_leaves(out):
      val = getattr(leaf, "value", leaf)
      self.assertIsInstance(val, jax.ShapeDtypeStruct)
    wi = getattr(out["decoder"]["layers_0"]["moe_block"]["wi"], "value", out["decoder"]["layers_0"]["moe_block"]["wi"])
    self.assertEqual(wi.shape, (EXPERTS, EMB, 32))

  def test_case_5_host_memory_profiling(self):
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    q_non_stream = ctx.Queue()
    p_non_stream = ctx.Process(target=_profile_conversion_worker, args=(False, q_non_stream))
    p_non_stream.start()
    delta_non_stream = q_non_stream.get(timeout=60)
    p_non_stream.join()

    q_stream = ctx.Queue()
    p_stream = ctx.Process(target=_profile_conversion_worker, args=(True, q_stream))
    p_stream.start()
    delta_stream = q_stream.get(timeout=60)
    p_stream.join()

    logging.info(
        "test_case_5_host_memory_profiling: delta_non_stream=%d KB, delta_stream=%d KB",
        delta_non_stream,
        delta_stream,
    )
    self.assertLessEqual(delta_stream, delta_non_stream)

  def test_case_6_parity_vs_raiden_unscan_on_homogeneous(self):
    from maxtext.integration.tunix.weight_mapping import raiden_unscan

    cfg = pytypes.SimpleNamespace(
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
    for k in base_flat:
      v_base = getattr(base_flat[k], "value", base_flat[k])
      v_conv = getattr(conv_flat[k], "value", conv_flat[k])
      self.assertEqual(v_base.shape, v_conv.shape, f"Shape mismatch at {k}")
      self.assertEqual(v_base.dtype, v_conv.dtype, f"Dtype mismatch at {k}")
      np.testing.assert_array_equal(np.asarray(v_base), np.asarray(v_conv), err_msg=f"Value mismatch at {k}")

  def test_case_7_streaming_piece_count_and_parity(self):
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    pieces = list(converter.convert_streaming(source, target_state=None, groups_per_piece=1))
    self.assertEqual(len(pieces), len(converter._direct._groups))

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
    for k in expected_flat:
      v_exp = getattr(expected_flat[k], "value", expected_flat[k])
      v_got = getattr(merged_flat[k], "value", merged_flat[k])
      self.assertEqual(v_exp.shape, v_got.shape, f"Shape mismatch at {k}")
      self.assertEqual(v_exp.dtype, v_got.dtype, f"Dtype mismatch at {k}")
      np.testing.assert_array_equal(np.asarray(v_exp), np.asarray(v_got), err_msg=f"Value mismatch at {k}")

  def test_case_8_streaming_piece_batching(self):
    cfg = _config(
        inhomogeneous_layer_cycle_interval=CYCLE,
        num_decoder_layers=NUM_LAYERS,
        prefuse_moe_weights=True,
    )
    source = _source_tree(True)
    converter = WeightConverter(config=cfg, rollout_backend="maxtext")
    pieces = list(converter.convert_streaming(source, target_state=None, groups_per_piece=2))
    num_groups = len(converter._direct._groups)
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
    for k in expected_flat:
      v_exp = getattr(expected_flat[k], "value", expected_flat[k])
      v_got = getattr(merged_flat[k], "value", merged_flat[k])
      np.testing.assert_array_equal(np.asarray(v_exp), np.asarray(v_got))

  def test_case_9_weight_converter_convert_streaming_dispatch(self):
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


if __name__ == "__main__":
  unittest.main()

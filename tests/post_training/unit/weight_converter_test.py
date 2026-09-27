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
import gc
import resource

# Must precede the first JAX import: the cross-mesh tests below need more than
# one CPU device, and the backend reads this only at initialization.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import types as pytypes  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

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
  """Returns a mock configuration object for tests."""
  cfg = pytypes.SimpleNamespace(
      num_decoder_layers=NUM_LAYERS,
      inhomogeneous_layer_cycle_interval=CYCLE,
      param_scan_axis=SCAN_AXIS,
      model_name="test_model",
      decoder_block="default",
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


def _profile_conversion_worker(is_streaming, result_queue):
  """Worker process to profile memory usage during conversion."""
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
    for k in expected_flat:
      v_exp = getattr(expected_flat[k], "value", expected_flat[k])
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
    for k in expected_flat:
      v_exp = getattr(expected_flat[k], "value", expected_flat[k])
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


def _fp32_leaf_source(float32_gate_logits: bool, logits_dot_in_fp32: bool):
  """Trainer state with the leaf dtypes MaxText gives it under these flags.

  Kernels and the embedding follow weight_dtype (bf16). The flags move specific
  leaves to float32, mirroring the model code: common_types.get_weight_dtype
  pins norms, GDN conv1d/A_log/dt_bias, the routed and shared-expert gates and
  logits_dense under float32_gate_logits; nnx_decoders pins decoder_norm and
  logits_dense under logits_dot_in_fp32. decoder_norm is *not* float32 under
  float32_gate_logits alone.

  Slot 0 is a GatedDeltaNet layer and slot 3 the full-attention layer of the
  shrunk qwen3.5 cycle. Layer leaves are scanned and take the "slice" op; the
  embedding, decoder_norm and logits_dense take "identity".
  """
  bf16 = jnp.bfloat16
  gated = jnp.float32 if float32_gate_logits else bf16
  logits = jnp.float32 if (float32_gate_logits or logits_dot_in_fp32) else bf16
  final_norm = jnp.float32 if logits_dot_in_fp32 else bf16

  def scanned(dtype, *shape):
    # +0.1 is not representable in bf16, so a float32 leaf rounded through bf16
    # on the way shows up as a value mismatch, not only as a dtype one.
    return _scanned(*shape, offset=0.1).astype(dtype)

  def layer(attention):
    return {
        "input_layernorm": {"scale": scanned(gated, EMB)},
        "post_attention_layernorm": {"scale": scanned(gated, EMB)},
        "attention": attention,
        "mlp": {
            "routed_experts": {"gate": {"kernel": scanned(gated, EMB, EXPERTS)}},
            "shared_expert": {"wi_0": {"kernel": scanned(bf16, EMB, MOE_DIM)}},
            "shared_expert_gate": {"kernel": scanned(gated, EMB, 1)},
        },
    }

  gdn = {
      "in_proj_qkvz": {"kernel": scanned(bf16, EMB, 12)},
      "conv1d": {"kernel": scanned(gated, 4, 1, 12)},
      "A_log": scanned(gated, 2),
      "dt_bias": scanned(gated, 2),
  }
  full_attention = {"attention": {"query": {"kernel": scanned(bf16, EMB, 2, 4)}}}
  return {
      "base": {
          "token_embedder": {"embedding": _arr(16, EMB).astype(bf16)},
          "decoder": {
              "layers": {"layer_0": layer(gdn), f"layer_{CYCLE - 1}": layer(full_attention)},
              "decoder_norm": {"scale": _arr(EMB, offset=0.1).astype(final_norm)},
              "logits_dense": {"kernel": _arr(EMB, 16, offset=0.1).astype(logits)},
          },
      }
  }


def _source_dtypes(source):
  return {".".join(key): jnp.dtype(value.dtype) for key, value in traverse_util.flatten_dict(source).items()}


def _dtypes_by_source(out):
  """{source path: dtype} for a target-free result, keyed back to its source.

  `layers_{i}` folds back to the scanned `layers.layer_{i % CYCLE}` it was
  sliced from, so a source leaf whose blocks disagree on dtype shows up as a set
  rather than hiding behind whichever block was read last.
  """
  seen = {}
  for key, leaf in traverse_util.flatten_dict(out).items():
    parts = []
    for part in key:
      if isinstance(part, str) and part.startswith("layers_"):
        parts += ["layers", f"layer_{int(part[len('layers_'):]) % CYCLE}"]
      else:
        parts.append(str(part))
    seen.setdefault(".".join(parts), set()).add(jnp.dtype(getattr(leaf, "value", leaf).dtype))
  return {path: next(iter(dtypes)) if len(dtypes) == 1 else dtypes for path, dtypes in seen.items()}


class TargetFreeFloat32LeafDtypeTest(unittest.TestCase):
  """Target-free sync must send each leaf in the dtype the rollout holds it in.

  With no target state the converter alone picks the wire dtype, and Tunix's
  transport preflight fails the whole sync round on any per-key item_size
  mismatch. A rollout built with float32_gate_logits or logits_dot_in_fp32
  holds the leaves those flags move to float32 in float32, so they must arrive
  float32; bf16 kernels must still arrive bf16, and with both flags off the old
  gate/router-only rule must be unchanged.
  """

  BF16 = jnp.dtype(jnp.bfloat16)
  F32 = jnp.dtype(jnp.float32)
  # The only paths the flag-independent rule lets keep their source dtype.
  GATES = frozenset(
      f"base.decoder.layers.layer_{slot}.mlp.{name}.kernel"
      for slot in (0, CYCLE - 1)
      for name in ("routed_experts.gate", "shared_expert_gate")
  )

  @staticmethod
  def _converter(**flags):
    return MaxTextToMaxTextConverter(_config(weight_dtype="bfloat16", **flags))

  def test_float32_gate_logits_keeps_every_float32_leaf(self):
    for logits_dot_in_fp32 in (False, True):
      source = _fp32_leaf_source(float32_gate_logits=True, logits_dot_in_fp32=logits_dot_in_fp32)
      # Sources here are only bf16 or float32, so "float32 stays float32, bf16
      # goes to target_dtype" is exactly "every leaf keeps its source dtype".
      want = _source_dtypes(source)
      # Guard the fixture itself: a float32 leaf of every kind the flag moves,
      # on both op paths, or this test could pass without exercising them.
      fp32 = {path for path, dtype in want.items() if dtype == self.F32}
      gdn = "base.decoder.layers.layer_0"
      self.assertLessEqual(
          {
              f"{gdn}.input_layernorm.scale",
              f"{gdn}.attention.conv1d.kernel",
              f"{gdn}.attention.A_log",
              f"{gdn}.attention.dt_bias",
              "base.decoder.logits_dense.kernel",
          }
          | self.GATES,
          fp32,
      )
      for abstract in (False, True):
        with self.subTest(logits_dot_in_fp32=logits_dot_in_fp32, abstract=abstract):
          src = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), source) if abstract else source
          converter = self._converter(float32_gate_logits=True, logits_dot_in_fp32=logits_dot_in_fp32)
          self.assertEqual(_dtypes_by_source(converter.convert(src, target_state=None)), want)
          self.assertEqual({g.op for g in converter._groups}, {"identity", "slice"})  # pylint: disable=protected-access

  def test_kept_float32_leaf_is_not_rounded_through_bf16(self):
    source = _fp32_leaf_source(float32_gate_logits=True, logits_dot_in_fp32=False)
    out = self._converter(float32_gate_logits=True).convert(source, target_state=None)["base"]["decoder"]
    src = source["base"]["decoder"]
    # Slice path: slot 0, block 1 is layers_4.
    got = getattr(out[f"layers_{CYCLE}"]["attention"]["A_log"], "value", out[f"layers_{CYCLE}"]["attention"]["A_log"])
    want = jnp.take(src["layers"]["layer_0"]["attention"]["A_log"], 1, axis=SCAN_AXIS)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # Identity path.
    got = getattr(out["logits_dense"]["kernel"], "value", out["logits_dense"]["kernel"])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(src["logits_dense"]["kernel"]))

  def test_flags_off_keeps_only_gate_and_router_dtypes(self):
    """With both flags off only gate/router leaves keep their source dtype."""
    source = _fp32_leaf_source(float32_gate_logits=True, logits_dot_in_fp32=True)
    want = {path: dtype if path in self.GATES else self.BF16 for path, dtype in _source_dtypes(source).items()}
    f32_leaf = jnp.zeros((2,), jnp.float32)
    # An absent flag must behave like an explicit False.
    for flags in ({}, {"float32_gate_logits": False, "logits_dot_in_fp32": False}):
      with self.subTest(flags=flags):
        converter = self._converter(**flags)
        arrived = _dtypes_by_source(converter.convert(source, target_state=None))
        self.assertEqual({path for path, dtype in arrived.items() if dtype == self.F32}, self.GATES)
        self.assertEqual(arrived, want)
        # No qwen3.5 param path contains "router", so the fixture cannot reach
        # that half of the rule; check it directly.
        self.assertEqual(
            converter._target_free_dtype(f32_leaf, "decoder.layers.layer_0.mlp.router.kernel", jnp.bfloat16),  # pylint: disable=protected-access
            self.F32,
        )

  def test_logits_dot_in_fp32_alone_keeps_decoder_norm_and_logits_dense(self):
    source = _fp32_leaf_source(float32_gate_logits=False, logits_dot_in_fp32=True)
    arrived = _dtypes_by_source(self._converter(logits_dot_in_fp32=True).convert(source, target_state=None))
    self.assertEqual(arrived["base.decoder.decoder_norm.scale"], self.F32)
    self.assertEqual(arrived["base.decoder.logits_dense.kernel"], self.F32)
    self.assertEqual(arrived, _source_dtypes(source))

  def test_bf16_leaf_is_never_upcast(self):
    """The rule keys on the source dtype, so an all-bf16 trainer stays bf16.

    Covers gate paths too, and both flags on, where an over-broad rule keyed on
    the path or the flag alone would up-cast.
    """
    source = _fp32_leaf_source(float32_gate_logits=False, logits_dot_in_fp32=False)
    for float32_gate_logits in (False, True):
      for logits_dot_in_fp32 in (False, True):
        with self.subTest(float32_gate_logits=float32_gate_logits, logits_dot_in_fp32=logits_dot_in_fp32):
          converter = self._converter(float32_gate_logits=float32_gate_logits, logits_dot_in_fp32=logits_dot_in_fp32)
          arrived = _dtypes_by_source(converter.convert(source, target_state=None))
          self.assertEqual(set(arrived), set(_source_dtypes(source)))
          self.assertEqual(set(arrived.values()), {self.BF16})

  def test_leaf_without_dtype_falls_back_to_target_dtype(self):
    converter = self._converter(float32_gate_logits=True, logits_dot_in_fp32=True)
    for path in ("decoder.decoder_norm.scale", "decoder.layers.layer_0.mlp.routed_experts.gate.kernel"):
      with self.subTest(path=path):
        self.assertEqual(
            converter._target_free_dtype(0.5, path, jnp.bfloat16), jnp.bfloat16  # pylint: disable=protected-access
        )


if __name__ == "__main__":
  unittest.main()

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

"""Unit tests for the pipeline parallelism helpers in maxtext.utils.pipeline_utils."""

import unittest

from absl.testing import parameterized
from flax import nnx
import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.utils import pipeline_utils

pytestmark = pytest.mark.cpu_only


def _decoder_layer_specs():
  """Physical specs shaped like a Mixtral-style MoE decoder layer, with a leading 'stage' dimension."""
  return {
      "MoeBlock_0": {"wi_0": P("stage", "expert", "fsdp", "tensor")},
      "self_attention": {"query": P("stage", "fsdp", "tensor")},
      "pre_self_attention_layer_norm": {"scale": P("stage", "tensor")},
  }


class GetMeshAxisDimIndicesTest(unittest.TestCase):
  """Tests for get_mesh_axis_dim_indices."""

  def test_finds_dimension_sharded_over_axis(self):
    specs = {"kernel": P("stage", "fsdp", "tensor"), "bias": P("stage", "tensor")}
    result = pipeline_utils.get_mesh_axis_dim_indices(specs, "fsdp")
    self.assertEqual(result, {"kernel": 1, "bias": -1})

  def test_defaults_to_fsdp_axis(self):
    specs = {"kernel": P("stage", "fsdp", "tensor")}
    self.assertEqual(pipeline_utils.get_mesh_axis_dim_indices(specs), {"kernel": 1})

  def test_finds_axis_inside_compound_entry(self):
    specs = {"kernel": P("stage", ("fsdp", "tensor"), None)}
    self.assertEqual(pipeline_utils.get_mesh_axis_dim_indices(specs, "tensor"), {"kernel": 1})

  def test_unsharded_spec_returns_minus_one(self):
    specs = {"kernel": P(None, None)}
    self.assertEqual(pipeline_utils.get_mesh_axis_dim_indices(specs, "fsdp"), {"kernel": -1})

  def test_preserves_tree_structure(self):
    specs = {"a": {"b": P("fsdp")}, "c": [P(None, "fsdp")]}
    result = pipeline_utils.get_mesh_axis_dim_indices(specs, "fsdp")
    self.assertEqual(result, {"a": {"b": 0}, "c": [1]})


class RemoveGatheredMeshAxesTest(parameterized.TestCase):
  """Tests for remove_gathered_mesh_axes."""

  def test_replaces_removed_axes_with_none(self):
    result = pipeline_utils.remove_gathered_mesh_axes(
        P("stage", "fsdp", "tensor"), is_moe_block_0=True, axes_to_remove=["fsdp"]
    )
    self.assertEqual(result, P("stage", None, "tensor"))

  def test_keeps_unlisted_axes_and_none_entries(self):
    result = pipeline_utils.remove_gathered_mesh_axes(
        P("stage", None, "tensor"), is_moe_block_0=True, axes_to_remove=["fsdp"]
    )
    self.assertEqual(result, P("stage", None, "tensor"))

  def test_expert_axis_removed_outside_moe_block(self):
    result = pipeline_utils.remove_gathered_mesh_axes(
        P("stage", "expert", "tensor"), is_moe_block_0=False, axes_to_remove=["fsdp"]
    )
    self.assertEqual(result, P("stage", None, "tensor"))

  def test_expert_axis_kept_in_moe_block(self):
    result = pipeline_utils.remove_gathered_mesh_axes(
        P("stage", "expert", "tensor"), is_moe_block_0=True, axes_to_remove=["fsdp"]
    )
    self.assertEqual(result, P("stage", "expert", "tensor"))

  def test_compound_entry_keeps_members_that_are_not_removed(self):
    result = pipeline_utils.remove_gathered_mesh_axes(
        P(("fsdp", "tensor"), None), is_moe_block_0=True, axes_to_remove=["fsdp"]
    )
    self.assertEqual(result, P(("tensor",), None))

  @parameterized.named_parameters(("moe_block", True), ("non_moe_block", False))
  def test_does_not_modify_axes_to_remove(self, is_moe_block_0):
    axes_to_remove = ["fsdp", "fsdp_transpose", "context"]
    pipeline_utils.remove_gathered_mesh_axes(
        P("stage", "expert", "fsdp"), is_moe_block_0=is_moe_block_0, axes_to_remove=axes_to_remove
    )
    self.assertEqual(axes_to_remove, ["fsdp", "fsdp_transpose", "context"])

  def test_non_partition_spec_is_returned_unchanged(self):
    sentinel = object()
    self.assertIs(pipeline_utils.remove_gathered_mesh_axes(sentinel, False, ["fsdp"]), sentinel)
    self.assertIsNone(pipeline_utils.remove_gathered_mesh_axes(None, False, ["fsdp"]))

  def test_unsupported_axis_type_raises(self):
    with self.assertRaisesRegex(ValueError, "Unsupported_axis_type"):
      pipeline_utils.remove_gathered_mesh_axes(P(1), is_moe_block_0=True, axes_to_remove=["fsdp"])


class DeriveStageWeightPartitionSpecsTest(parameterized.TestCase):
  """Tests for derive_stage_weight_partition_specs."""

  def test_drops_leading_dimension_and_gathered_axes(self):
    specs = {"kernel": P("stage", "fsdp", "tensor"), "scale": P("stage", "tensor")}
    result = pipeline_utils.derive_stage_weight_partition_specs(specs, ["fsdp", "context"])
    self.assertEqual(result, {"kernel": P(None, "tensor"), "scale": P("tensor")})

  def test_expert_axis_removed_from_non_moe_leaves_only(self):
    specs = {
        "MoeBlock_0": {"wi_0": P("stage", "expert", "fsdp", "tensor")},
        "mlp": {"wi_0": P("stage", "expert", "fsdp", "tensor")},
    }
    result = pipeline_utils.derive_stage_weight_partition_specs(specs, ["fsdp"])
    self.assertEqual(result["MoeBlock_0"]["wi_0"], P("expert", None, "tensor"))
    self.assertEqual(result["mlp"]["wi_0"], P(None, None, "tensor"))

  def test_moe_block_keeps_expert_axis_in_every_sublayer_of_a_stage(self):
    # With num_layers_per_pipeline_stage > 1 a stage is an NNXSequentialPipelineStage whose children are named
    # layers_0, layers_1, ... JAX visits layers_0's non-MoE leaves before layers_1's MoeBlock_0, so state leaking
    # between leaves would wrongly strip 'expert' from layers_1's routed experts.
    specs = {f"layers_{i}": _decoder_layer_specs() for i in range(3)}
    result = pipeline_utils.derive_stage_weight_partition_specs(specs, ["fsdp", "fsdp_transpose", "context"])
    for i in range(3):
      with self.subTest(layer=i):
        self.assertEqual(result[f"layers_{i}"]["MoeBlock_0"]["wi_0"], P("expert", None, "tensor"))
        self.assertEqual(result[f"layers_{i}"]["self_attention"]["query"], P(None, "tensor"))

  @parameterized.named_parameters(
      ("sibling_sorts_before", "Attention"),
      ("sibling_sorts_after", "self_attention"),
  )
  def test_moe_result_does_not_depend_on_sibling_traversal_order(self, sibling_name):
    specs = {
        "MoeBlock_0": {"wi_0": P("stage", "expert", "fsdp", "tensor")},
        sibling_name: {"query": P("stage", "fsdp", "tensor")},
    }
    result = pipeline_utils.derive_stage_weight_partition_specs(specs, ["fsdp"])
    self.assertEqual(result["MoeBlock_0"]["wi_0"], P("expert", None, "tensor"))

  def test_does_not_modify_axes_to_remove(self):
    axes_to_remove = ["fsdp", "fsdp_transpose", "context"]
    pipeline_utils.derive_stage_weight_partition_specs({"layers_0": _decoder_layer_specs()}, axes_to_remove)
    self.assertEqual(axes_to_remove, ["fsdp", "fsdp_transpose", "context"])


class StripPipelineRepeatLogicalAxisTest(unittest.TestCase):
  """Tests for strip_pipeline_repeat_logical_axis."""

  def test_removes_circular_repeats_from_every_spec(self):
    specs = {
        "kernel": P("circular_repeats", "layers", "embed"),
        "scale": P("layers", "circular_repeats", "norm"),
    }
    result = pipeline_utils.strip_pipeline_repeat_logical_axis(specs)
    self.assertEqual(result, {"kernel": P("layers", "embed"), "scale": P("layers", "norm")})

  def test_keeps_other_entries_in_order_including_none(self):
    result = pipeline_utils.strip_pipeline_repeat_logical_axis({"kernel": P("layers", None, "embed")})
    self.assertEqual(result, {"kernel": P("layers", None, "embed")})

  def test_none_spec_returns_none(self):
    self.assertIsNone(pipeline_utils.strip_pipeline_repeat_logical_axis(None))


class IsSpecLeafTest(unittest.TestCase):
  """Tests for is_spec_leaf."""

  def test_partition_spec_and_none_are_leaves(self):
    self.assertTrue(pipeline_utils.is_spec_leaf(P("fsdp", None)))
    self.assertTrue(pipeline_utils.is_spec_leaf(P()))
    self.assertTrue(pipeline_utils.is_spec_leaf(None))

  def test_other_values_are_not_leaves(self):
    for value in ("fsdp", ("fsdp", None), {"a": P("fsdp")}, [P("fsdp")], 0):
      with self.subTest(value=value):
        self.assertFalse(pipeline_utils.is_spec_leaf(value))

  def test_treats_none_as_a_leaf_when_used_with_tree_map(self):
    specs = {"a": P("fsdp"), "b": None}
    result = jax.tree.map(lambda p: "leaf", specs, is_leaf=pipeline_utils.is_spec_leaf)
    self.assertEqual(result, {"a": "leaf", "b": "leaf"})


class IsStaticParamTest(unittest.TestCase):
  """Tests for is_static_param."""

  def test_matches_nnx_param(self):
    self.assertTrue(pipeline_utils.is_static_param((), nnx.Param(jnp.ones(2))))

  def test_matches_fp8_overwrite_with_gradient_variable_by_class_name(self):
    fp8_variable_cls = type("_overwrite_with_gradient", (), {})
    self.assertTrue(pipeline_utils.is_static_param((), fp8_variable_cls()))

  def test_rejects_non_param_variables(self):
    self.assertFalse(pipeline_utils.is_static_param((), nnx.BatchStat(jnp.ones(2))))
    self.assertFalse(pipeline_utils.is_static_param((), nnx.Intermediate(jnp.ones(2))))
    self.assertFalse(pipeline_utils.is_static_param((), jnp.ones(2)))


class NnxStateFlattenTest(unittest.TestCase):
  """Tests for flatten_nnx_state / unflatten_nnx_state."""

  @staticmethod
  def _variables(state):
    return jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))

  def test_round_trip_preserves_values_variable_types_and_structure(self):
    state = nnx.state(nnx.BatchNorm(3, rngs=nnx.Rngs(0)))
    flat = pipeline_utils.flatten_nnx_state(state)
    restored = pipeline_utils.unflatten_nnx_state(*flat)

    self.assertEqual(jax.tree.structure(restored), jax.tree.structure(state))
    original_vars, restored_vars = self._variables(state), self._variables(restored)
    self.assertEqual([type(v) for v in restored_vars], [type(v) for v in original_vars])
    for original, new in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
      np.testing.assert_array_equal(new, original)

  def test_flatten_reports_variable_flags_types_and_raw_arrays(self):
    state = nnx.state(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))
    arrays, _, is_var_flags, var_types, var_metadata = pipeline_utils.flatten_nnx_state(state)

    self.assertEqual(len(arrays), 2)
    self.assertTrue(all(is_var_flags))
    self.assertEqual(var_types, [nnx.Param, nnx.Param])
    self.assertEqual(len(var_metadata), 2)
    for array in arrays:
      self.assertNotIsInstance(array, nnx.Variable)

  def test_round_trip_preserves_variable_metadata(self):
    state = {"w": nnx.Param(jnp.ones((2, 2)), my_tag="kept")}
    restored = pipeline_utils.unflatten_nnx_state(*pipeline_utils.flatten_nnx_state(state))
    self.assertEqual(restored["w"].get_metadata("my_tag"), "kept")

  def test_plain_array_leaves_pass_through(self):
    state = {"a": jnp.arange(3)}
    arrays, treedef, is_var_flags, var_types, var_metadata = pipeline_utils.flatten_nnx_state(state)

    self.assertEqual(is_var_flags, [False])
    self.assertEqual(var_types, [None])
    self.assertEqual(var_metadata, [{}])
    restored = pipeline_utils.unflatten_nnx_state(arrays, treedef, is_var_flags, var_types, var_metadata)
    np.testing.assert_array_equal(restored["a"], state["a"])


class AdvanceRngStateTest(unittest.TestCase):
  """Tests for advance_rng_state."""

  @staticmethod
  def _rng_state():
    class _Module(nnx.Module):

      def __init__(self, rngs):
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    return nnx.state(_Module(nnx.Rngs(0)), nnx.RngState)

  @staticmethod
  def _variables_by_type(state, variable_type):
    leaves = jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
    return [v for v in leaves if isinstance(v, variable_type)]

  def test_folds_iteration_into_every_rng_key(self):
    state = self._rng_state()
    advanced = pipeline_utils.advance_rng_state(state, 3)

    old_keys = self._variables_by_type(state, nnx.RngKey)
    new_keys = self._variables_by_type(advanced, nnx.RngKey)
    self.assertGreater(len(old_keys), 0)
    for old, new in zip(old_keys, new_keys):
      expected = jax.random.fold_in(old[...], 3)
      np.testing.assert_array_equal(jax.random.key_data(new[...]), jax.random.key_data(expected))

  def test_leaves_rng_counters_untouched(self):
    state = self._rng_state()
    advanced = pipeline_utils.advance_rng_state(state, 3)

    old_counts = self._variables_by_type(state, nnx.RngCount)
    new_counts = self._variables_by_type(advanced, nnx.RngCount)
    self.assertGreater(len(old_counts), 0)
    for old, new in zip(old_counts, new_counts):
      self.assertEqual(new[...].dtype, jnp.uint32)
      np.testing.assert_array_equal(new[...], old[...])

  def test_different_iterations_give_different_keys(self):
    state = self._rng_state()

    def key_data_at(iteration):
      advanced = pipeline_utils.advance_rng_state(state, iteration)
      return [jax.random.key_data(v[...]) for v in self._variables_by_type(advanced, nnx.RngKey)]

    for key_0, key_1 in zip(key_data_at(0), key_data_at(1)):
      self.assertFalse(np.array_equal(key_0, key_1))

  def test_does_not_modify_input_state(self):
    state = self._rng_state()
    before = [jax.random.key_data(v[...]) for v in self._variables_by_type(state, nnx.RngKey)]
    pipeline_utils.advance_rng_state(state, 5)
    after = [jax.random.key_data(v[...]) for v in self._variables_by_type(state, nnx.RngKey)]
    for old, new in zip(before, after):
      np.testing.assert_array_equal(new, old)

  def test_folds_each_key_of_a_stacked_key_array(self):
    stacked = jax.random.split(jax.random.key(0), 4)
    state = {"rngs": {"key": nnx.RngKey(stacked, tag="default")}}
    advanced = pipeline_utils.advance_rng_state(state, 7)

    expected = jax.vmap(lambda k: jax.random.fold_in(k, 7))(stacked)
    np.testing.assert_array_equal(jax.random.key_data(advanced["rngs"]["key"][...]), jax.random.key_data(expected))


class LinenCollectionConversionTest(unittest.TestCase):
  """Tests for arrays_to_linen_collection / linen_collection_to_arrays."""

  def test_round_trip_preserves_order(self):
    keys = ["w", "b", "scale"]
    arrays = [jnp.zeros(1), jnp.ones(2), jnp.full((3,), 2.0)]
    collection = pipeline_utils.arrays_to_linen_collection(arrays, keys)

    self.assertEqual(list(collection), keys)
    restored = pipeline_utils.linen_collection_to_arrays(collection, keys)
    self.assertEqual(len(restored), len(arrays))
    for original, new in zip(arrays, restored):
      self.assertIs(new, original)

  def test_extraction_follows_requested_key_order(self):
    collection = {"a": 1, "b": 2, "c": 3}
    self.assertEqual(pipeline_utils.linen_collection_to_arrays(collection, ["c", "a"]), [3, 1])


class PipelineContextTest(unittest.TestCase):
  """Tests for PipelineContext."""

  def test_captures_pipeline_methods_and_is_not_a_pytree(self):
    class _FakePipeline:
      """Exposes the four attributes PipelineContext captures from a pipeline module."""

      weight_prefetching = True

      def run_one_iteration(self):
        return "run"

      def from_all_variables_to_repeat_weights(self):
        return "repeat"

      def from_repeat_weights_to_bsw(self):
        return "bsw"

    pipeline = _FakePipeline()
    context = pipeline_utils.PipelineContext(pipeline)

    self.assertTrue(context.weight_prefetching)
    self.assertEqual(context.run_one_iteration(), "run")
    self.assertEqual(context.from_all_variables_to_repeat_weights(), "repeat")
    self.assertEqual(context.from_repeat_weights_to_bsw(), "bsw")
    # JAX must treat the context as an opaque leaf so it never tries to flatten captured attributes.
    self.assertEqual(jax.tree.leaves(context), [context])


class GradientAccumulationScanTest(unittest.TestCase):
  """create_gradient_accumulation_scan must match the gradients of a plain jax.lax.scan over the same step."""

  class _FakeModel:
    """Stands in for the pipeline module: one microbatch step and its remat policy."""

    def run_one_iteration(
        self, lightweight_state, bsw, positions, segment_ids, deterministic, model_mode, logical_partition_spec=None
    ):
      del positions, segment_ids, deterministic, model_mode, logical_partition_spec
      return jnp.tanh(lightweight_state @ bsw["w"] + bsw["b"])

    def get_pipeline_remat_policy(self):
      return jax.checkpoint_policies.nothing_saveable

  def setUp(self):
    super().setUp()
    key_state, key_w, key_b = jax.random.split(jax.random.key(0), 3)
    self.length = 4
    self.model = self._FakeModel()
    self.loop_state = jax.random.normal(key_state, (2, 3))
    self.bsw = {"w": 0.5 * jax.random.normal(key_w, (3, 3)), "b": 0.1 * jax.random.normal(key_b, (3,))}
    self.positions = jnp.arange(3)
    self.segment_ids = jnp.ones((3,), dtype=jnp.int32)

  def _custom_scan(self, loop_state, bsw):
    scan = pipeline_utils.create_gradient_accumulation_scan(self.model, self.length)
    return scan(loop_state, bsw, self.positions, self.segment_ids)

  def _reference_scan(self, loop_state, bsw):
    def step(carry, _):
      return self.model.run_one_iteration(carry, bsw, self.positions, self.segment_ids, True, None), None

    final, _ = jax.lax.scan(step, loop_state, None, length=self.length)
    return final, bsw

  @staticmethod
  def _loss(outputs):
    final, bsw_out = outputs
    # The bsw passthrough term checks that its cotangent is added to the accumulated scan gradient.
    return jnp.sum(final**2) + 3.0 * jnp.sum(bsw_out["w"]) + jnp.sum(bsw_out["b"] ** 2)

  def test_forward_matches_reference_scan(self):
    final, bsw_out = self._custom_scan(self.loop_state, self.bsw)
    expected_final, _ = self._reference_scan(self.loop_state, self.bsw)

    np.testing.assert_allclose(final, expected_final, rtol=1e-6, atol=1e-6)
    jax.tree.map(np.testing.assert_array_equal, bsw_out, self.bsw)

  def test_gradients_match_reference_scan(self):
    custom_grads = jax.grad(lambda s, b: self._loss(self._custom_scan(s, b)), argnums=(0, 1))(self.loop_state, self.bsw)
    reference_grads = jax.grad(lambda s, b: self._loss(self._reference_scan(s, b)), argnums=(0, 1))(
        self.loop_state, self.bsw
    )

    jax.tree.map(
        lambda got, want: np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5), custom_grads, reference_grads
    )

  def test_gradients_survive_jit(self):
    grad_fn = jax.grad(lambda s, b: self._loss(self._custom_scan(s, b)), argnums=(0, 1))
    eager = grad_fn(self.loop_state, self.bsw)
    jitted = jax.jit(grad_fn)(self.loop_state, self.bsw)

    jax.tree.map(lambda got, want: np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5), jitted, eager)


if __name__ == "__main__":
  unittest.main()

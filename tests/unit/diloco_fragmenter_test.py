# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the streaming DiLoCo FragmentedTreeManipulator."""

import functools
import re
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import parameterized
import drjax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from maxtext.configs import types
from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco.utils import fragmenter
from maxtext.trainers.diloco.utils import spmd_diloco_sync
from maxtext.trainers.diloco.utils.fragmenter import BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX

_NUM_LAYERS = 4
_VOCAB = 11  # Not divisible by the number of layer fragments: exercises the remainder rows.
_EMB = 6


def _config(num_fragments, bucketize, sequential=False):
  return SimpleNamespace(
      num_decoder_layers=_NUM_LAYERS,
      num_diloco_fragments=num_fragments,
      use_sequential_layers=sequential,
      param_scan_axis=0,
      diloco_bucketize_non_scanned=bucketize,
  )


def _params(replicas=None):
  """A tiny decoder-shaped tree whose values encode their own position, optionally with a leading replica dim."""

  def arange(*shape):
    shape = shape if replicas is None else (replicas, *shape)
    return jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

  return {
      "decoder": {
          "layers": {"mlp": arange(_NUM_LAYERS, _EMB, 3), "norm": arange(_NUM_LAYERS, _EMB)},
          "final_norm": arange(_EMB),
          "logits_dense": {"kernel": arange(_EMB, _VOCAB)},  # (embed, vocab), like the MaxText output head.
      },
      "token_embedder": {"embedding": arange(_VOCAB, _EMB)},
  }


def _keystr(*path):
  return jax.tree_util.keystr(tuple(jax.tree_util.DictKey(k) for k in path))


def _shardings(mesh, embedding_spec, head_spec):
  """NamedShardings with the structure of `_params()`; everything but the embedding table and head is replicated."""
  rep = NamedSharding(mesh, P())
  return {
      "decoder": {
          "layers": {"mlp": rep, "norm": rep},
          "final_norm": rep,
          "logits_dense": {"kernel": NamedSharding(mesh, head_spec)},
      },
      "token_embedder": {"embedding": NamedSharding(mesh, embedding_spec)},
  }


class FragmentedTreeManipulatorTest(parameterized.TestCase):

  @parameterized.product(bucketize=[False, True], sequential=[False, True], num_fragments=[2, 3, 5])
  def test_round_trip_is_identity(self, bucketize, sequential, num_fragments):
    params = _params()
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(num_fragments, bucketize, sequential))
    tree = params
    for f in range(num_fragments):
      tree = manipulator.apply_flat_fragment(tree, f, manipulator.get_flat_fragment(tree, f))
    jax.tree.map(np.testing.assert_array_equal, tree, params)

  @parameterized.product(bucketize=[False, True], sequential=[False, True], num_fragments=[2, 3, 5])
  def test_fragments_partition_every_parameter_exactly_once(self, bucketize, sequential, num_fragments):
    """Adding 1 to each fragment in turn must leave every element of every leaf at exactly 1 (no gap, no overlap)."""
    params = _params()
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(num_fragments, bucketize, sequential))
    counts = jax.tree.map(jnp.zeros_like, params)
    for f in range(num_fragments):
      frag = manipulator.get_flat_fragment(counts, f)
      counts = manipulator.apply_flat_fragment(counts, f, {k: v + 1 for k, v in frag.items()})
    for path, leaf in jax.tree_util.tree_flatten_with_path(counts)[0]:
      np.testing.assert_array_equal(leaf, np.ones(leaf.shape), err_msg=jax.tree_util.keystr(path))

  @parameterized.parameters(False, True)
  def test_fragment_owners_of_embedding_and_head(self, bucketize):
    num_fragments = 3
    params = _params()
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(num_fragments, bucketize))
    marks = jax.tree.map(jnp.zeros_like, params)
    for f in range(num_fragments):
      frag = manipulator.get_flat_fragment(marks, f)
      marks = manipulator.apply_flat_fragment(marks, f, {k: v + f + 1 for k, v in frag.items()})
    expected_embedding_owner = np.full((_VOCAB, _EMB), 1.0)  # Fragment 0 owns the whole table without bucketing.
    if bucketize:
      chunk = _VOCAB // (num_fragments - 1)
      expected_embedding_owner = np.concatenate(
          [np.full((chunk, _EMB), f + 1.0) for f in range(1, num_fragments)]
          + [np.full((_VOCAB - chunk * (num_fragments - 1), _EMB), 1.0)]
      )
    np.testing.assert_array_equal(marks["token_embedder"]["embedding"], expected_embedding_owner)
    # The head (embed, vocab) is split along vocab too, so its owners are the embedding's, transposed.
    np.testing.assert_array_equal(marks["decoder"]["logits_dense"]["kernel"], expected_embedding_owner.T)

  def test_bucketized_keys_and_shapes(self):
    num_fragments = 3  # 2 layer fragments -> 5 vocab indices each, 1 remainder.
    params = _params()
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(num_fragments, bucketize=True))
    embedding = _keystr("token_embedder", "embedding")
    head = _keystr("decoder", "logits_dense", "kernel")
    self.assertEqual(
        manipulator.bucketized_leaves,
        {
            embedding: fragmenter.BucketSpec(chunk_size=5, remainder=1, axis=0),
            head: fragmenter.BucketSpec(chunk_size=5, remainder=1, axis=1),
        },
    )

    frag0 = manipulator.get_flat_fragment(params, 0)
    self.assertEqual(
        set(frag0),
        {_keystr("decoder", "final_norm"), embedding + REMAINDER_KEY_SUFFIX, head + REMAINDER_KEY_SUFFIX},
    )
    np.testing.assert_array_equal(frag0[embedding + REMAINDER_KEY_SUFFIX], params["token_embedder"]["embedding"][10:])
    np.testing.assert_array_equal(frag0[head + REMAINDER_KEY_SUFFIX], params["decoder"]["logits_dense"]["kernel"][:, 10:])

    frag2 = manipulator.get_flat_fragment(params, 2)
    np.testing.assert_array_equal(frag2[embedding + BUCKET_KEY_SUFFIX], params["token_embedder"]["embedding"][5:10])
    np.testing.assert_array_equal(frag2[head + BUCKET_KEY_SUFFIX], params["decoder"]["logits_dense"]["kernel"][:, 5:10])
    self.assertEqual(frag2[_keystr("decoder", "layers", "mlp")].shape, (2, _EMB, 3))

  @parameterized.parameters(True, False)
  def test_logs_each_bucketized_leaf(self, bucketize):
    with mock.patch.object(fragmenter.max_logging, "log") as log:
      fragmenter.FragmentedTreeManipulator.create(_params(), _config(3, bucketize=bucketize))
    lines = [c.args[0] for c in log.call_args_list if c.args[0].startswith("DiLoCo: bucketizing")]
    if not bucketize:
      self.assertEmpty(lines)
      return
    self.assertLen(lines, 2)
    head = [l for l in lines if l.startswith(f"DiLoCo: bucketizing {_keystr('decoder', 'logits_dense', 'kernel')} ")]
    self.assertLen(head, 1)
    self.assertIn("BucketSpec(chunk_size=5, remainder=1, axis=1)", head[0])

  def test_bucketization_is_off_by_default(self):
    self.assertFalse(types.DilocoParams.model_fields["diloco_bucketize_non_scanned"].default)

  def test_replica_dim(self):
    num_fragments, replicas = 3, 2
    params = _params()
    replicated = _params(replicas=replicas)
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(num_fragments, bucketize=True))
    for f in range(num_fragments):
      frag = manipulator.get_flat_fragment(replicated, f, has_replica_dim=True)
      per_replica = manipulator.get_flat_fragment(params, f)
      self.assertEqual(set(frag), set(per_replica))
      for k, v in frag.items():
        self.assertEqual(v.shape, (replicas, *per_replica[k].shape))
      restored = manipulator.apply_flat_fragment(replicated, f, frag, has_replica_dim=True)
      jax.tree.map(np.testing.assert_array_equal, restored, replicated)

  def test_shape_dtype_struct_leaves(self):
    params = _params()
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    manipulator = fragmenter.FragmentedTreeManipulator.create(abstract, _config(3, bucketize=True))
    for f in range(3):
      frag = manipulator.get_flat_fragment(abstract, f)
      concrete = manipulator.get_flat_fragment(params, f)
      self.assertEqual({k: v.shape for k, v in frag.items()}, {k: v.shape for k, v in concrete.items()})
      restored = manipulator.apply_flat_fragment(abstract, f, frag)
      self.assertEqual(jax.tree.structure(restored), jax.tree.structure(abstract))

  @parameterized.named_parameters(
      # MaxText's default rules shard vocab over `tensor` and the embedding dim (`embed_vocab`) over `fsdp`.
      ("fsdp_only", (8, 1), P("tensor", "fsdp"), P("fsdp", "tensor"), {"embedding": 0, "head": 1}),
      ("tensor_parallel", (4, 2), P("tensor", "fsdp"), P("fsdp", "tensor"), {}),
      ("vocab_sharded_embed_unsharded", (4, 2), P("tensor", None), P(None, "tensor"), {"embedding": 1, "head": 0}),
      ("unconstrained_is_never_split", (8, 1), P(P.UNCONSTRAINED, None), P(None, None), {"embedding": 1, "head": 1}),
  )
  def test_split_axis_is_an_unsharded_axis(self, mesh_shape, embedding_spec, head_spec, expected_axes):
    mesh = jax.sharding.AbstractMesh(mesh_shape, ("fsdp", "tensor"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    shardings = _shardings(mesh, embedding_spec, head_spec)
    params = _params()
    with mock.patch.object(fragmenter.max_logging, "log") as log:
      manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(3, bucketize=True), shardings=shardings)
    self.assertEqual(
        {("embedding" if "embedding" in k else "head"): spec.axis for k, spec in manipulator.bucketized_leaves.items()},
        expected_axes,
    )
    skipped = [c.args[0] for c in log.call_args_list if c.args[0].startswith("DiLoCo: not bucketizing")]
    self.assertLen(skipped, 2 - len(expected_axes))
    # Whatever the split, every element still belongs to exactly one fragment.
    rebuilt = jax.tree.map(jnp.zeros_like, params)
    for f in range(3):
      rebuilt = manipulator.apply_flat_fragment(rebuilt, f, manipulator.get_flat_fragment(params, f))
    jax.tree.map(np.testing.assert_array_equal, rebuilt, params)

  def test_uses_shardings_of_concrete_leaves(self):
    if jax.device_count() < 2:
      self.skipTest("Needs 2 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=8).")
    mesh = Mesh(np.array(jax.devices()[:2]).reshape((1, 2)), ("fsdp", "tensor"))
    params = _params()
    params["token_embedder"]["embedding"] = jnp.zeros((12, _EMB))  # Even vocab so that it can be sharded 2 ways.
    params["decoder"]["logits_dense"]["kernel"] = jnp.zeros((_EMB, 12))
    shardings = _shardings(mesh, P("tensor", None), P(None, "tensor"))
    placed = jax.device_put(params, shardings)
    config = _config(3, bucketize=True)
    from_leaves = fragmenter.FragmentedTreeManipulator.create(placed, config).bucketized_leaves
    explicit = fragmenter.FragmentedTreeManipulator.create(params, config, shardings=shardings).bucketized_leaves
    self.assertEqual(from_leaves, explicit)
    self.assertEqual({spec.axis for spec in from_leaves.values()}, {0, 1})
    self.assertEqual(from_leaves[_keystr("token_embedder", "embedding")].axis, 1)  # vocab (axis 0) is sharded

  def test_shardings_must_match_the_tree(self):
    with self.assertRaisesRegex(ValueError, "shardings has 1 leaves"):
      fragmenter.FragmentedTreeManipulator.create(_params(), _config(3, bucketize=True), shardings=[None])

  def test_shardings_must_match_the_tree_structure(self):
    """Same number of leaves under different keys is rejected, not matched up by position."""
    mesh = jax.sharding.AbstractMesh((8, 1), ("fsdp", "tensor"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    shardings = _shardings(mesh, P("tensor", "fsdp"), P("fsdp", "tensor"))
    shardings["zz_embedder"] = shardings.pop("token_embedder")
    with self.assertRaisesRegex(ValueError, "does not have the structure of the parameter tree"):
      fragmenter.FragmentedTreeManipulator.create(_params(), _config(3, bucketize=True), shardings=shardings)

  def test_warns_without_any_named_sharding(self):
    mesh = jax.sharding.AbstractMesh((8, 1), ("fsdp", "tensor"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    shardings = _shardings(mesh, P("tensor", "fsdp"), P("fsdp", "tensor"))
    for kwargs, expected_warnings in (
        ({}, 1),
        ({"shardings": jax.tree.map(lambda _: None, shardings)}, 1),
        ({"shardings": shardings}, 0),
    ):
      with mock.patch.object(fragmenter.max_logging, "warning") as warning:
        fragmenter.FragmentedTreeManipulator.create(_params(), _config(3, bucketize=True), **kwargs)
      self.assertEqual(warning.call_count, expected_warnings, kwargs.keys())

  @parameterized.parameters(False, True)
  def test_create_on_tracers(self, with_shardings):
    """Inside a trace the leaves carry no sharding: explicit shardings give the eager split, otherwise all unsharded."""
    mesh = jax.sharding.AbstractMesh((4, 2), ("fsdp", "tensor"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    # Vocab sharded over `tensor`, embed unsharded: the split axis is embed with shardings and vocab without.
    shardings = _shardings(mesh, P("tensor", None), P(None, "tensor")) if with_shardings else None
    params = _params()
    config = _config(3, bucketize=True)
    traced = {}

    def build(p):
      traced["manipulator"] = fragmenter.FragmentedTreeManipulator.create(p, config, shardings=shardings)
      return p

    with mock.patch.object(fragmenter.max_logging, "warning") as warning:
      jax.jit(build).lower(params)
    self.assertEqual(warning.call_count, 0 if with_shardings else 1)
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    eager = fragmenter.FragmentedTreeManipulator.create(abstract, config, shardings=shardings)
    self.assertEqual(traced["manipulator"].bucketized_leaves, eager.bucketized_leaves)
    expected_axis = 1 if with_shardings else 0
    self.assertEqual(traced["manipulator"].bucketized_leaves[_keystr("token_embedder", "embedding")].axis, expected_axis)

  @parameterized.product(bucketize=[False, True], sequential=[False, True])
  def test_apply_casts_to_the_tree_dtype(self, bucketize, sequential):
    """A fragment in another dtype (f32) is written into a bf16 tree on every path, keeping the tree's dtype."""
    params = _params()
    tree = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(3, bucketize, sequential))
    for f in range(1, 3):  # Layer fragments: scanned leaves, plus buckets when bucketized.
      restored = manipulator.apply_flat_fragment(tree, f, manipulator.get_flat_fragment(params, f))
      for path, leaf in jax.tree_util.tree_flatten_with_path(restored)[0]:
        self.assertEqual(leaf.dtype, jnp.bfloat16, jax.tree_util.keystr(path))
      jax.tree.map(np.testing.assert_array_equal, restored, tree)

  def test_fragment_idx_out_of_range(self):
    params = _params()
    manipulator = fragmenter.FragmentedTreeManipulator.create(params, _config(3, bucketize=True))
    with self.assertRaisesRegex(ValueError, r"fragment_idx \(3\) must be in \[0, 3\)"):
      manipulator.get_flat_fragment(params, 3)
    with self.assertRaisesRegex(ValueError, r"fragment_idx \(-1\) must be in \[0, 3\)"):
      manipulator.apply_flat_fragment(params, -1, {})


class SpmdStreamingSyncTest(parameterized.TestCase):
  """SPMD streaming DiLoCo syncs fragments through the same manipulator as threaded DiLoCo."""

  @parameterized.parameters(False, True)
  def test_syncing_every_fragment_once_equals_one_full_outer_step(self, bucketize):
    # The outer step is elementwise and every element belongs to exactly one fragment, so one pass over all fragments
    # (with the inner params held fixed) must match one outer step on the whole tree.
    num_fragments, replicas = 3, 2
    outer = _params()
    inner = jax.tree.map(lambda x: jnp.stack([0.5 * x, 0.25 * x + 1.0]), outer)
    outer_optimizer = optax.sgd(0.7, momentum=0.9, nesterov=True)
    manipulator = fragmenter.FragmentedTreeManipulator.create(outer, _config(num_fragments, bucketize))

    @drjax.program(placements={"diloco": replicas})
    def sync_all_fragments(params, inner_params, opt_state):
      state = diloco.DiLoCoTrainState(
          inner_state=SimpleNamespace(model=nnx.State(jax.tree.map(nnx.Param, inner_params))),
          params=params,
          outer_opt_state=opt_state,
          step=0,
      )
      for f in range(num_fragments):
        state = spmd_diloco_sync.synchronize_fragment_state(state, manipulator, f, outer_optimizer)
      return state.params, state.outer_opt_state

    params, opt_state = sync_all_fragments(outer, inner, outer_optimizer.init(outer))

    pseudo_grad = jax.tree.map(lambda o, i: o - i.mean(axis=0), outer, inner)
    updates, expected_opt_state = outer_optimizer.update(pseudo_grad, outer_optimizer.init(outer), outer)
    jax.tree.map(np.testing.assert_allclose, params, optax.apply_updates(outer, updates))
    jax.tree.map(np.testing.assert_allclose, opt_state[0].trace, expected_opt_state[0].trace)

  def test_tensor_parallel_sync_adds_no_all_gather(self):
    """With vocab sharded over `tensor`, slicing it would make XLA gather whole leaves; the fragmenter must avoid it."""
    if jax.device_count() < 8:
      self.skipTest("Needs 8 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=8).")
    mesh = Mesh(np.array(jax.devices()[:8]).reshape((2, 2, 2)), ("diloco", "fsdp", "tensor"))
    params = _params()
    params["token_embedder"]["embedding"] = jnp.ones((12, _EMB))
    params["decoder"]["logits_dense"]["kernel"] = jnp.ones((_EMB, 12))
    outer_shardings = _shardings(mesh, P("tensor", "fsdp"), P("fsdp", "tensor"))
    inner_shardings = jax.tree.map(lambda s: NamedSharding(mesh, P("diloco", *s.spec)), outer_shardings)
    outer = jax.device_put(params, outer_shardings)
    inner = jax.device_put(jax.tree.map(lambda x: jnp.stack([x, 2 * x]), params), inner_shardings)
    optimizer = optax.sgd(0.1, momentum=0.9, nesterov=True)
    opt_state = jax.device_put(optimizer.init(outer), (optax.TraceState(trace=outer_shardings), optax.EmptyState()))

    def resharding_collectives(shardings):
      manipulator = fragmenter.FragmentedTreeManipulator.create(outer, _config(3, True), shardings=shardings)

      @drjax.program(placements={"diloco": 2})
      def sync_and_apply(params, inner_params, opt_state):
        state = diloco.DiLoCoTrainState(
            inner_state=nnx.State({"model": jax.tree.map(nnx.Param, inner_params)}),
            params=params,
            outer_opt_state=opt_state,
            step=0,
        )
        state = spmd_diloco_sync.synchronize_fragment_state(state, manipulator, 1, optimizer, mesh=mesh)
        state = spmd_diloco_sync.apply_fragment_to_inner_state(state, manipulator, 1, mesh=mesh)
        return state.params, state.outer_opt_state, nnx.State(state.inner_state["model"]).to_pure_dict()

      out_shardings = (outer_shardings, (optax.TraceState(trace=outer_shardings), optax.EmptyState()), inner_shardings)
      hlo = jax.jit(sync_and_apply, out_shardings=out_shardings).lower(outer, inner, opt_state).compile().as_text()
      return _count_resharding_collectives(hlo)

    self.assertEqual(resharding_collectives(outer_shardings), 0)
    # Control: splitting the sharded vocab axis (no sharding information) does gather, so the check above is sensitive.
    self.assertGreater(resharding_collectives(jax.tree.map(lambda _: None, outer_shardings)), 0)


def _count_resharding_collectives(hlo: str) -> int:
  """Counts the collectives that slicing a sharded axis produced in the pre-fix HLO (gathers and permutes)."""
  return len(re.findall(r" (?:all-gather|collective-permute|all-to-all)(?:-start)?\(", hlo))


class SpmdStreamingTrainStepTest(parameterized.TestCase):
  """Runs `diloco.build_diloco_train_step` (SPMD streaming, bucketized) on an 8-device CPU mesh."""

  _REPLICAS = 2
  _NUM_FRAGMENTS = 3
  # Embedding (vocab, embed) and head (embed, vocab) layouts. FSDP-only leaves vocab unsharded, so both leaves are
  # bucketized along vocab; with tensor parallelism every long axis is sharded, so neither is bucketized.
  _LAYOUTS = {
      "fsdp": (P(None, "fsdp"), P("fsdp", None)),
      "tensor_parallel": (P("tensor", "fsdp"), P("fsdp", "tensor")),
  }

  def setUp(self):
    """Builds a (diloco 2, fsdp 2, tensor 2) mesh, a streaming config with one fragment per step, and params."""
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest("Needs 8 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=8).")
    self.mesh = Mesh(np.array(jax.devices()[:8]).reshape((2, 2, 2)), ("diloco", "fsdp", "tensor"))
    self.config = SimpleNamespace(
        **vars(_config(self._NUM_FRAGMENTS, bucketize=True)),
        diloco_outer_lr=0.7,
        diloco_outer_momentum=0.9,
        diloco_sync_period=self._NUM_FRAGMENTS,  # One fragment per step.
        num_communication_overlapping_steps=0,
        communication_overlapping_alpha=0.0,
        num_diloco_replicas=self._REPLICAS,
        enable_streaming_diloco=True,
    )
    self.outer_optimizer = optax.sgd(0.7, momentum=0.9, nesterov=True)
    params = _params()
    params["token_embedder"]["embedding"] = jnp.arange(12 * _EMB, dtype=jnp.float32).reshape(12, _EMB) / 7
    params["decoder"]["logits_dense"]["kernel"] = jnp.arange(_EMB * 12, dtype=jnp.float32).reshape(_EMB, 12) / 5
    self.params = params
    # Each replica's toy inner step adds its own batch value to every parameter.
    self.batch = jax.device_put(jnp.array([1.0, 3.0]), NamedSharding(self.mesh, P("diloco")))

  def _state(self, outer_shardings):
    mesh = self.mesh
    inner_shardings = jax.tree.map(lambda s: NamedSharding(mesh, P("diloco", *s.spec)), outer_shardings)
    inner_model = jax.device_put(jax.tree.map(lambda x: jnp.stack([x] * self._REPLICAS), self.params), inner_shardings)
    step = jax.device_put(jnp.zeros((self._REPLICAS,), jnp.int32), NamedSharding(mesh, P("diloco")))
    outer = jax.device_put(self.params, outer_shardings)
    opt_state = jax.device_put(
        self.outer_optimizer.init(outer), (optax.TraceState(trace=outer_shardings), optax.EmptyState())
    )
    return diloco.DiLoCoTrainState(
        inner_state=nnx.State({"model": jax.tree.map(nnx.Param, inner_model), "optimizer": {"step": step}}),
        params=outer,
        outer_opt_state=opt_state,
        step=jnp.int32(0),
    )

  @staticmethod
  def _toy_inner_step(state, batch, rng):
    del rng
    model = jax.tree.map(lambda p: p + batch, state["model"])
    return nnx.State({"model": model, "optimizer": {"step": state["optimizer"]["step"] + 1}}), batch

  def _train_step(self, outer_shardings):
    return diloco.build_diloco_train_step(
        self.config, self._toy_inner_step, mesh=self.mesh, outer_params_shardings=outer_shardings
    )

  @parameterized.named_parameters(("fsdp", "fsdp"), ("tensor_parallel", "tensor_parallel"))
  def test_compiled_step_has_no_resharding_collectives(self, layout):
    outer_shardings = _shardings(self.mesh, *self._LAYOUTS[layout])
    state = self._state(outer_shardings)
    hlo = jax.jit(self._train_step(outer_shardings)).lower(state, self.batch, None).compile().as_text()
    self.assertEqual(_count_resharding_collectives(hlo), 0)
    if layout == "tensor_parallel":
      # Control: without sharding information the sharded vocab axis is split, which gathers.
      no_shardings = jax.tree.map(lambda _: None, outer_shardings)
      hlo = jax.jit(self._train_step(no_shardings)).lower(state, self.batch, None).compile().as_text()
      self.assertGreater(_count_resharding_collectives(hlo), 0)

  @parameterized.named_parameters(("fsdp", "fsdp"), ("tensor_parallel", "tensor_parallel"))
  def test_each_step_syncs_and_applies_one_fragment(self, layout):
    """Every step must match a host-side outer step on that step's fragment, with alpha=0 and no delay."""
    outer_shardings = _shardings(self.mesh, *self._LAYOUTS[layout])
    manipulator = fragmenter.FragmentedTreeManipulator.create(self.params, self.config, shardings=outer_shardings)
    self.assertLen(manipulator.bucketized_leaves, 2 if layout == "fsdp" else 0)
    train_step = jax.jit(self._train_step(outer_shardings))
    state = self._state(outer_shardings)
    batch = np.asarray(self.batch)
    assert_close = functools.partial(np.testing.assert_allclose, rtol=1e-6, atol=1e-6)
    for step in range(1, 2 * self._NUM_FRAGMENTS + 1):
      # Host copies as (single-device) jax arrays: the interleaved-layer path writes with `.at[]`.
      outer = jax.tree.map(jnp.asarray, jax.device_get(state.params))
      trace = jax.tree.map(jnp.asarray, jax.device_get(state.outer_opt_state[0].trace))
      inner = jax.tree.map(jnp.asarray, jax.device_get(nnx.State(state.inner_state["model"]).to_pure_dict()))
      inner = jax.tree.map(lambda x: x + batch.reshape((-1,) + (1,) * (x.ndim - 1)), inner)  # The inner step.

      f = step % self._NUM_FRAGMENTS
      outer_frag = manipulator.get_flat_fragment(outer, f)
      inner_frag = manipulator.get_flat_fragment(inner, f, has_replica_dim=True)
      pseudo_grad = jax.tree.map(lambda o, i: o - i.mean(axis=0), outer_frag, inner_frag)
      trace_frag = (optax.TraceState(trace=manipulator.get_flat_fragment(trace, f)), optax.EmptyState())
      updates, new_trace_frag = self.outer_optimizer.update(pseudo_grad, trace_frag, outer_frag)
      new_outer_frag = optax.apply_updates(outer_frag, updates)
      expected_outer = manipulator.apply_flat_fragment(outer, f, new_outer_frag)
      expected_trace = manipulator.apply_flat_fragment(trace, f, new_trace_frag[0].trace)
      replicated_frag = jax.tree.map(lambda x: jnp.stack([x] * self._REPLICAS), new_outer_frag)
      expected_inner = manipulator.apply_flat_fragment(inner, f, replicated_frag, has_replica_dim=True)

      state, _ = train_step(state, self.batch, None)
      self.assertEqual(int(state.step), step)
      jax.tree.map(assert_close, jax.device_get(state.params), expected_outer)
      jax.tree.map(assert_close, jax.device_get(state.outer_opt_state[0].trace), expected_trace)
      jax.tree.map(assert_close, jax.device_get(nnx.State(state.inner_state["model"]).to_pure_dict()), expected_inner)

  def test_bucketization_requires_outer_params_shardings(self):
    with self.assertRaisesRegex(ValueError, "diloco_bucketize_non_scanned requires outer_params_shardings"):
      self._train_step(None)


if __name__ == "__main__":
  unittest.main()

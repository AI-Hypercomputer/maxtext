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

"""Unit tests for threaded streaming DiLoCo fragment transfer utilities."""

import re
from types import SimpleNamespace
import unittest

from absl.testing import parameterized
import drjax
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco.utils import fragment_transfer
from maxtext.trainers.diloco.utils import spmd_diloco_sync
from maxtext.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator
from maxtext.utils.diloco_sharding import split_mesh_along_axis

_NUM_LAYERS = 4
# GitHub CI runs the unit tests on 4 CPU devices. Every mesh below uses exactly the first 4 devices, so the transfer
# layouts the tests assert are the same with 4 or more devices.
_NUM_DEVICES = 4


def _skip_unless_devices(test, num_devices=_NUM_DEVICES):
  if jax.device_count() < num_devices:
    test.skipTest(f"Needs {num_devices} devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count={num_devices}).")


def _devices():
  return np.array(jax.devices()[:_NUM_DEVICES])


def _mesh():
  return Mesh(_devices(), ("fsdp",))


def _params(mesh, seed=0):
  """Scanned stacks sharded on a trailing dim, a replicated vector and two bucketizable tables."""
  keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8))

  def leaf(shape, spec):
    return jax.device_put(jax.random.normal(next(keys), shape), NamedSharding(mesh, spec))

  return {
      "decoder": {
          "layers": {"mlp": leaf((_NUM_LAYERS, 4, 16), P(None, None, "fsdp")), "norm": leaf((_NUM_LAYERS, 4), P())},
          "final_norm": leaf((16,), P("fsdp")),
          # (embed, vocab) with embed over fsdp, like MaxText's logits_dense. Split along vocab (the longest axis),
          # every slice keeps whole fsdp shards; an embed-axis split (10 rows with 3 fragments) would not split
          # over 4 fsdp shards.
          "logits_dense": {"kernel": leaf((20, 44), P("fsdp", None))},
      },
      "token_embedder": {"embedding": leaf((11, 16), P(None, "fsdp"))},
      "logits": {"kernel": leaf((11, 12), P(None, None))},  # Replicated: flattened with a global reshape.
  }


def _multi_axis_params(mesh, seed=0, dtype=jnp.float32):
  """Leaves with tuple PartitionSpecs over a ('data', 'fsdp', 'tensor') or ('fsdp', 'tensor') mesh."""
  keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8))
  data = ("data",) if "data" in mesh.axis_names else ()

  def leaf(shape, spec):
    return jax.device_put(jax.random.normal(next(keys), shape).astype(dtype), NamedSharding(mesh, spec))

  return {
      "decoder": {
          "layers": {
              "mlp": leaf((_NUM_LAYERS, 8, 8), P(None, (*data, "fsdp"), "tensor")),
              # Scan axis sharded over 'tensor': a one-layer fragment (1, 8) does not split evenly, so it falls back
              # to a global reshape; a two-layer fragment does split.
              "attn": leaf((_NUM_LAYERS, 8), P("tensor", "fsdp")),
          },
          "final_norm": leaf((8,), P()),
          "logits_dense": {"kernel": leaf((8, 12), P((*data, "fsdp"), None))},
      },
      "token_embedder": {"embedding": leaf((12, 8), P(None, ("fsdp", "tensor")))},
  }


def _manipulator(params, num_fragments=3, sequential=False, bucketize=True):
  config = SimpleNamespace(
      num_decoder_layers=_NUM_LAYERS,
      num_diloco_fragments=num_fragments,
      use_sequential_layers=sequential,
      param_scan_axis=0,
      diloco_bucketize_non_scanned=bucketize,
  )
  return FragmentedTreeManipulator.create(params, config)


def _copy(tree):
  return jax.tree.map(jnp.copy, tree)


class FragmentTransferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _skip_unless_devices(self)
    self.mesh = _mesh()

  def _assert_round_trip(self, transfer, manipulator, source, target):
    """Moves every fragment of `source` into `target` and checks each step against the manipulator."""
    for f in range(manipulator.num_fragments):
      # `apply` donates its input, and untouched leaves of `reference` alias `target`: compare against copies.
      reference = manipulator.apply_flat_fragment(_copy(target), f, manipulator.get_flat_fragment(source, f))
      target = transfer.apply(target, f, transfer.extract(source, f))
      jax.tree.map(np.testing.assert_array_equal, target, reference)
    jax.tree.map(np.testing.assert_array_equal, target, source)

  @parameterized.product(sequential=[False, True], bucketize=[False, True], num_fragments=[2, 3, 5])
  def test_extract_then_apply_moves_exactly_the_fragment(self, sequential, bucketize, num_fragments):
    source = _params(self.mesh, seed=0)
    manipulator = _manipulator(source, num_fragments, sequential, bucketize)
    transfer = fragment_transfer.FragmentTransfer(manipulator, source)
    self._assert_round_trip(transfer, manipulator, source, _params(self.mesh, seed=1))

  def test_transfer_arrays_are_1d_and_shard_local_when_possible(self):
    params = _params(self.mesh)
    transfer = fragment_transfer.FragmentTransfer(_manipulator(params), params)
    layer_fragment = transfer.extract(params, 1)
    for key, value in layer_fragment.items():
      self.assertEqual(value.ndim, 1, key)
    specs = {leaf.key: leaf.spec for leaf in transfer.layouts[1]}
    self.assertEqual(specs["['decoder']['layers']['mlp']"], P(None, None, "fsdp"))
    self.assertIsNone(specs["['decoder']['layers']['norm']"])  # Replicated: global reshape.
    self.assertEqual(layer_fragment["['decoder']['layers']['mlp']"].sharding.spec, P(("fsdp",)))
    self.assertEqual(specs["['decoder']['logits_dense']['kernel']#bucket"], P("fsdp", None))  # Vocab split: shard-local.
    # Only replicated leaves fall back to a global reshape; every sharded leaf splits evenly over the 4 fsdp shards.
    self.assertEqual(
        {k for k, s in specs.items() if s is None}, {"['decoder']['layers']['norm']", "['logits']['kernel']#bucket"}
    )
    self.assertEqual(
        {leaf.key: leaf.spec for leaf in transfer.layouts[0]},
        {"['decoder']['final_norm']": P("fsdp"), "['token_embedder']['embedding']#remainder": P(None, "fsdp")},
    )

  def test_apply_preserves_shardings_and_donates(self):
    params = _params(self.mesh)
    manipulator = _manipulator(params)
    transfer = fragment_transfer.FragmentTransfer(manipulator, params)
    source = _params(self.mesh, seed=3)
    expected_values = manipulator.apply_flat_fragment(_copy(params), 2, manipulator.get_flat_fragment(source, 2))
    expected_shardings = jax.tree.map(lambda x: x.sharding, params)
    updated = transfer.apply(params, 2, transfer.extract(source, 2))
    for leaf in jax.tree.leaves(params):
      self.assertTrue(leaf.is_deleted())
    jax.tree.map(lambda x, s: self.assertEqual(x.sharding, s), updated, expected_shardings)
    jax.tree.map(np.testing.assert_array_equal, updated, expected_values)

  def test_fragment_idx_out_of_range_is_rejected(self):
    params = _params(self.mesh)
    transfer = fragment_transfer.FragmentTransfer(_manipulator(params, num_fragments=3), params)
    fragment = transfer.extract(params, 2)
    for bad in (3, -1):
      with self.assertRaisesRegex(ValueError, re.escape(f"fragment_idx ({bad}) must be in [0, 3)")):
        transfer.extract(params, bad)
      with self.assertRaisesRegex(ValueError, re.escape(f"fragment_idx ({bad}) must be in [0, 3)")):
        transfer.apply(params, bad, fragment)

  def test_fragment_index_is_not_copied_between_devices(self):
    # A learner on devices other than the default one: a jnp index would be created on the default device and copied.
    mesh = Mesh(_devices()[2:4], ("fsdp",))
    self.assertNotIn(jax.devices()[0], mesh.devices.flat)
    params = _params(mesh)
    transfer = fragment_transfer.FragmentTransfer(_manipulator(params), params)
    fragment = transfer.extract(params, 1)
    with jax.transfer_guard_device_to_device("disallow"):
      fragment = transfer.extract(params, 1)
      transfer.apply(params, 1, fragment)

  # (data, fsdp, tensor) = (1, 2, 2): ('fsdp', 'tensor') tuples and the 'tensor'-sharded scan axis are real splits.
  # (2, 2, 1): ('data', 'fsdp') tuples are a real two-axis split.
  @parameterized.product(mesh_shape=[(1, 2, 2), (2, 2, 1)], num_fragments=[3, 5], dtype=[jnp.float32, jnp.bfloat16])
  def test_multi_axis_mesh_with_tuple_specs(self, mesh_shape, num_fragments, dtype):
    mesh = Mesh(_devices().reshape(mesh_shape), ("data", "fsdp", "tensor"))
    source = _multi_axis_params(mesh, seed=0, dtype=dtype)
    manipulator = _manipulator(source, num_fragments)
    transfer = fragment_transfer.FragmentTransfer(manipulator, source)
    specs = {leaf.key: leaf.spec for leaf in transfer.layouts[1]}
    self.assertEqual(specs["['decoder']['layers']['mlp']"], P(None, ("data", "fsdp"), "tensor"))
    self.assertEqual(specs["['token_embedder']['embedding']#bucket"], P(None, ("fsdp", "tensor")))
    fragment = transfer.extract(source, 1)
    self.assertEqual(fragment["['decoder']['layers']['mlp']"].sharding.spec, P(("data", "fsdp", "tensor")))
    self.assertEqual(fragment["['token_embedder']['embedding']#bucket"].sharding.spec, P(("fsdp", "tensor")))
    attn = fragment["['decoder']['layers']['attn']"]
    # One layer per fragment over 2 'tensor' shards: (1, 8) does not split over 'tensor', so a global reshape.
    attn_falls_back = num_fragments == 5 and mesh.shape["tensor"] == 2
    if attn_falls_back:
      self.assertIsNone(specs["['decoder']['layers']['attn']"])
      self.assertEqual(attn.sharding, NamedSharding(mesh, P()))
    else:
      self.assertEqual(attn.sharding.spec, P(("tensor", "fsdp")))
    # No other leaf of a layer fragment falls back; in fragment 0 only the replicated vector does.
    expected_fallbacks = {"['decoder']['layers']['attn']"} if attn_falls_back else set()
    self.assertEqual({k for k, s in specs.items() if s is None}, expected_fallbacks)
    self.assertEqual({leaf.key for leaf in transfer.layouts[0] if leaf.spec is None}, {"['decoder']['final_norm']"})
    for value in fragment.values():
      self.assertEqual(value.dtype, dtype)
    self._assert_round_trip(transfer, manipulator, source, _multi_axis_params(mesh, seed=1, dtype=dtype))

  # Two learners of 2 devices each, with either 'fsdp' or 'tensor' split.
  @parameterized.parameters({"mesh_shape": (2, 1, 2)}, {"mesh_shape": (2, 2, 1)})
  def test_round_trip_between_split_submeshes(self, mesh_shape):
    learner_meshes = split_mesh_along_axis(Mesh(_devices().reshape(mesh_shape), ("diloco", "fsdp", "tensor")), "diloco")
    source = _multi_axis_params(learner_meshes[0], seed=0)
    target = _multi_axis_params(learner_meshes[1], seed=1)
    manipulator = _manipulator(source, num_fragments=3)
    source_transfer = fragment_transfer.FragmentTransfer(manipulator, source)
    target_transfer = fragment_transfer.FragmentTransfer(manipulator, target)
    # Reference: the same values placed on the target submesh, so the manipulator never mixes device sets.
    source_on_target = jax.device_put(source, jax.tree.map(lambda x: x.sharding, target))
    for f in range(manipulator.num_fragments):
      reference = manipulator.apply_flat_fragment(_copy(target), f, manipulator.get_flat_fragment(source_on_target, f))
      moved = fragment_transfer.move_fragment(source_transfer.extract(source, f), learner_meshes[1])
      for value in moved.values():
        self.assertEqual(value.sharding.mesh, learner_meshes[1])
      target = target_transfer.apply(target, f, moved)
      jax.tree.map(np.testing.assert_array_equal, target, reference)
    jax.tree.map(np.testing.assert_array_equal, target, source)

  @parameterized.product(sequential=[False, True], num_fragments=[3, 5])
  def test_layer_fragments_share_one_executable(self, sequential, num_fragments):
    params = _params(self.mesh)
    manipulator = _manipulator(params, num_fragments=num_fragments, sequential=sequential)
    transfer = fragment_transfer.FragmentTransfer(manipulator, params)
    for f in range(1, num_fragments):
      params = transfer.apply(params, f, transfer.extract(params, f))
    self.assertEqual(transfer._extract_layer._cache_size(), 1)  # pylint: disable=protected-access
    self.assertEqual(transfer._apply_layer._cache_size(), 1)  # pylint: disable=protected-access

  def test_interleaved_layers_are_sliced_without_gather_or_scatter(self):
    params = _params(self.mesh)
    transfer = fragment_transfer.FragmentTransfer(_manipulator(params, num_fragments=3, sequential=False), params)
    self.assertEqual(transfer._layer_stride, 2)  # pylint: disable=protected-access
    index = np.int32(1)
    fragment = transfer.extract(params, 1)
    hlo = (
        transfer._extract_layer.lower(params, index).as_text()  # pylint: disable=protected-access
        + transfer._apply_layer.lower(params, index, fragment).as_text()  # pylint: disable=protected-access
    )
    self.assertNotRegex(hlo, r"gather|scatter")

  def test_shardings_must_be_named_and_on_one_mesh(self):
    params = _params(self.mesh)
    manipulator = _manipulator(params)
    unplaced = dict(params, logits={"kernel": jnp.ones((11, 12))})
    with self.assertRaisesRegex(ValueError, r"NamedSharding on every parameter; \['logits'\]\['kernel'\]"):
      fragment_transfer.FragmentTransfer(manipulator, unplaced)
    other_mesh = Mesh(_devices()[::-1], ("fsdp",))
    two_meshes = dict(
        params, logits={"kernel": jax.device_put(params["logits"]["kernel"], NamedSharding(other_mesh, P()))}
    )
    with self.assertRaisesRegex(ValueError, "one mesh; found 2 meshes"):
      fragment_transfer.FragmentTransfer(manipulator, two_meshes)

  # Fragment 0 and the layer fragments are applied by separate executables.
  @parameterized.parameters(0, 1)
  def test_alpha_interpolates_with_the_current_value(self, fragment_idx):
    current = _params(self.mesh, seed=0)
    synced = _params(self.mesh, seed=1)
    manipulator = _manipulator(current)
    alpha = 0.25
    transfer = fragment_transfer.FragmentTransfer(manipulator, current, alpha=alpha)
    reference = jax.tree.map(lambda c, s: alpha * c + (1 - alpha) * s, current, synced)
    expected = manipulator.get_flat_fragment(reference, fragment_idx)
    updated = transfer.apply(_copy(current), fragment_idx, transfer.extract(synced, fragment_idx))
    actual = manipulator.get_flat_fragment(updated, fragment_idx)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6), actual, expected)

  def test_same_layout_as(self):
    learner_meshes = split_mesh_along_axis(Mesh(_devices().reshape((2, 2, 1)), ("diloco", "fsdp", "tensor")), "diloco")
    learner_params = [_multi_axis_params(mesh) for mesh in learner_meshes]
    manipulator = _manipulator(learner_params[0])
    first, second = (fragment_transfer.FragmentTransfer(manipulator, p) for p in learner_params)
    self.assertTrue(first.same_layout_as(second))

    # The layout does not depend on which devices the mesh holds...
    params = _params(self.mesh)
    manipulator = _manipulator(params)
    transfer = fragment_transfer.FragmentTransfer(manipulator, params)
    reversed_mesh = Mesh(_devices()[::-1], ("fsdp",))
    on_reversed = jax.device_put(params, jax.tree.map(lambda x: NamedSharding(reversed_mesh, x.sharding.spec), params))
    self.assertTrue(transfer.same_layout_as(fragment_transfer.FragmentTransfer(manipulator, on_reversed)))
    # ...but it does on the mesh axis sizes: the same specs over 2 fsdp shards flatten in a different order.
    half_mesh = Mesh(_devices()[:2], ("fsdp",))
    on_half = jax.device_put(params, jax.tree.map(lambda x: NamedSharding(half_mesh, x.sharding.spec), params))
    self.assertFalse(transfer.same_layout_as(fragment_transfer.FragmentTransfer(manipulator, on_half)))
    # Different leaf layouts on the same mesh.
    self.assertFalse(transfer.same_layout_as(fragment_transfer.FragmentTransfer(_manipulator(params, 5), params)))

  def test_non_arithmetic_layer_assignment_is_rejected(self):
    params = _params(self.mesh)
    manipulator = _manipulator(params, num_fragments=3)
    manipulator.fragment_to_layer_indices = {1: (0, 3), 2: (1, 2)}
    with self.assertRaisesRegex(ValueError, "arithmetic progression"):
      fragment_transfer.FragmentTransfer(manipulator, params)

  def test_move_fragment_keeps_partition_specs(self):
    params = _params(self.mesh)
    transfer = fragment_transfer.FragmentTransfer(_manipulator(params), params)
    fragment = transfer.extract(params, 1)
    other_mesh = Mesh(_devices()[::-1], ("fsdp",))
    moved = fragment_transfer.move_fragment(fragment, other_mesh)
    for key, value in moved.items():
      self.assertEqual(value.sharding.mesh, other_mesh)
      self.assertEqual(value.sharding.spec, fragment[key].sharding.spec)
      np.testing.assert_array_equal(value, fragment[key])


class NesterovOuterStepTest(unittest.TestCase):

  def test_matches_optax_sgd_nesterov(self):
    lr, momentum = 0.7, 0.9
    rng = np.random.default_rng(0)
    outer = {"a": jnp.asarray(rng.normal(size=16), jnp.float32), "b": jnp.asarray(rng.normal(size=4), jnp.float32)}
    optimizer = optax.sgd(lr, momentum=momentum, nesterov=True)
    opt_state = optimizer.init(outer)
    reference = outer
    trace = jax.tree.map(jnp.zeros_like, outer)
    for _ in range(3):
      learners = [jax.tree.map(lambda x: x + jnp.asarray(rng.normal(size=x.shape), jnp.float32), outer) for _ in range(3)]
      mean = jax.tree.map(lambda *xs: sum(xs) / len(xs), *learners)
      updates, opt_state = optimizer.update(jax.tree.map(jnp.subtract, reference, mean), opt_state, reference)
      reference = optax.apply_updates(reference, updates)
      outer, trace = fragment_transfer.nesterov_outer_step(outer, trace, learners, learning_rate=lr, momentum=momentum)
      jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), outer, reference)

  def test_unchanged_elements_get_no_update(self):
    # With 3 learners that did not move, `outer - mean(learners)` is not exactly 0 for every value (3 * x / 3 rounds);
    # the mean of the differences is.
    outer = {"a": jnp.asarray(np.random.default_rng(0).normal(size=4096), jnp.float32)}
    trace = {"a": jnp.zeros(4096, jnp.float32)}
    new_outer, new_trace = fragment_transfer.nesterov_outer_step(
        outer, trace, [outer, outer, outer], learning_rate=0.7, momentum=0.9
    )
    np.testing.assert_array_equal(new_trace["a"], np.zeros(4096, np.float32))
    np.testing.assert_array_equal(new_outer["a"], outer["a"])

  def test_bfloat16_inputs_are_computed_in_float32(self):
    lr, momentum, n = 0.7, 0.9, 3
    rng = np.random.default_rng(0)
    outer = {"a": jnp.asarray(rng.normal(size=1024), jnp.bfloat16)}
    trace = {"a": jnp.asarray(0.1 * rng.normal(size=1024), jnp.bfloat16)}
    learners = [{"a": (outer["a"] + jnp.asarray(0.01 * rng.normal(size=1024), jnp.bfloat16))} for _ in range(n)]
    # Float32 reference computed from the same (bfloat16) inputs, rounded to bfloat16 once at the end.
    o32, t32 = np.asarray(outer["a"], np.float32), np.asarray(trace["a"], np.float32)
    pseudo_grad = sum(o32 - np.asarray(f["a"], np.float32) for f in learners) / np.float32(n)
    trace_ref = np.float32(momentum) * t32 + pseudo_grad
    outer_ref = o32 - np.float32(lr) * (pseudo_grad + np.float32(momentum) * trace_ref)
    new_outer, new_trace = fragment_transfer.nesterov_outer_step(
        outer, trace, learners, learning_rate=lr, momentum=momentum
    )
    self.assertEqual(new_outer["a"].dtype, jnp.bfloat16)
    self.assertEqual(new_trace["a"].dtype, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(new_outer["a"], np.float32), outer_ref.astype(jnp.bfloat16).astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(new_trace["a"], np.float32), trace_ref.astype(jnp.bfloat16).astype(np.float32)
    )

  def test_donates_trace_but_not_outer_or_learners(self):
    device = jax.devices()[0]
    outer = {"a": jax.device_put(jnp.ones(8), device)}
    trace = {"a": jax.device_put(jnp.zeros(8), device)}
    learners = [{"a": jax.device_put(jnp.full(8, 0.5), device)} for _ in range(2)]
    fragment_transfer.nesterov_outer_step(outer, trace, learners, learning_rate=0.1, momentum=0.9)
    self.assertTrue(trace["a"].is_deleted())
    self.assertFalse(outer["a"].is_deleted())
    self.assertFalse(any(f["a"].is_deleted() for f in learners))

  def test_outputs_keep_the_shardings_of_outer_and_trace(self):
    _skip_unless_devices(self)
    mesh = Mesh(_devices(), ("x",))
    replicated, sharded = NamedSharding(mesh, P()), NamedSharding(mesh, P("x"))
    outer = {"a": jax.device_put(jnp.ones(16), replicated)}
    trace = {"a": jax.device_put(jnp.zeros(16), replicated)}
    learners = [{"a": jax.device_put(jnp.full(16, 0.5), sharded)} for _ in range(2)]
    new_outer, new_trace = fragment_transfer.nesterov_outer_step(outer, trace, learners, learning_rate=0.1, momentum=0.9)
    self.assertEqual(new_outer["a"].sharding, replicated)
    self.assertEqual(new_trace["a"].sharding, replicated)


class SpmdParityTest(parameterized.TestCase):
  """The threaded outer step on transfer fragments must match SPMD streaming DiLoCo's `synchronize_fragment_state`."""

  @parameterized.parameters(2, 3)
  def test_threaded_outer_step_matches_spmd_fragment_sync(self, replicas):
    _skip_unless_devices(self)
    lr, momentum, num_fragments, rounds = 0.7, 0.9, 3, 2
    mesh = _mesh()
    outer = _params(mesh, seed=0)
    manipulator = _manipulator(outer, num_fragments)
    transfer = fragment_transfer.FragmentTransfer(manipulator, outer)
    optimizer = optax.sgd(lr, momentum=momentum, nesterov=True)

    @drjax.program(placements={"diloco": replicas})
    def spmd_sync(params, inner_params, opt_state):
      state = diloco.DiLoCoTrainState(
          inner_state=SimpleNamespace(model=nnx.State(jax.tree.map(nnx.Param, inner_params))),
          params=params,
          outer_opt_state=opt_state,
          step=0,
      )
      for f in range(num_fragments):
        state = spmd_diloco_sync.synchronize_fragment_state(state, manipulator, f, optimizer)
      return state.params, state.outer_opt_state

    spmd_params, spmd_opt_state = outer, optimizer.init(outer)
    threaded_outer = {f: transfer.extract(outer, f) for f in range(num_fragments)}
    threaded_trace = {
        f: {k: jnp.zeros(v.shape, v.dtype, device=v.sharding) for k, v in threaded_outer[f].items()}
        for f in range(num_fragments)
    }
    for r in range(rounds):
      learners = [
          jax.tree.map(lambda x, d: x + 0.1 * d, outer, _params(mesh, seed=100 * r + i + 1)) for i in range(replicas)
      ]
      spmd_params, spmd_opt_state = spmd_sync(
          spmd_params, jax.tree.map(lambda *xs: jnp.stack(xs), *learners), spmd_opt_state
      )
      for f in range(num_fragments):
        threaded_outer[f], threaded_trace[f] = fragment_transfer.nesterov_outer_step(
            threaded_outer[f],
            threaded_trace[f],
            [transfer.extract(learner, f) for learner in learners],
            learning_rate=lr,
            momentum=momentum,
        )

    threaded_params, threaded_full_trace = _copy(outer), jax.tree.map(jnp.zeros_like, outer)
    for f in range(num_fragments):
      threaded_params = transfer.apply(threaded_params, f, threaded_outer[f])
      threaded_full_trace = transfer.apply(threaded_full_trace, f, threaded_trace[f])

    def close(a, b):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    jax.tree.map(close, threaded_params, spmd_params)
    jax.tree.map(close, threaded_full_trace, spmd_opt_state[0].trace)


if __name__ == "__main__":
  unittest.main()

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

"""Tests for `maxtext.m3.core.sharding`.

Sharding bugs are the quiet kind: a wrong rule still trains, just slower, and
no assertion fires. So these tests lean on exact `PartitionSpec` equality
rather than on "it ran".

Device-count note: the mesh arithmetic lives in `_resolve_axis_sizes`, which
takes its target as a plain integer and is therefore fully testable on a
single CPU. Only the tests that need a real multi-device mesh are marked
`tpu_only`. That marker matters -- the repo conftest auto-marks unmarked tests
`cpu_only` and skips them on accelerator testbeds, so a multi-device test
without it would silently never run anywhere it matters.

Slice-count note: no testbed here spans more than one slice, and a device's
slice membership comes from an attribute jax sets, so the multi-slice tests
stub the slice count and jax's placement algorithm. What survives that
stubbing is the part m3 actually owns -- that ICI resolves against
devices-per-slice while DCN resolves against slices -- and the arithmetic
itself is covered for real, on CPU, through `_resolve_axis_sizes`.
"""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.m3.core import sharding as sharding_lib
from maxtext.m3.core.sharding import MESH_AXIS_NAMES
from maxtext.m3.core.sharding import Sharding
from maxtext.m3.core.sharding import TensorType
from maxtext.m3.core.sharding import _ambient_mesh_is_explicit
from maxtext.m3.core.sharding import _count_slices
from maxtext.m3.core.sharding import _resolve_axis_sizes
from maxtext.m3.core.sharding import create_mesh


class _ExampleSharding(Sharding):
  """A representative rule set, shaped like one a real model would ship."""

  def map_axis(self, axis, tensor_name, tensor_type):
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("dp", "fsdp")
      case "embed", TensorType.WEIGHT:
        return "fsdp"
      case "embed", TensorType.ACTIVATION:
        return None
      case ("heads" | "kv_heads"), _:
        return "tp"
      case "head_dim", _:
        return None
      case "mlp", _:
        return "tp"
      case "length", _:
        return None
      case _:
        raise KeyError(f"unmapped logical axis {axis!r} on tensor {tensor_name!r}")


class _BadSharding(Sharding):
  """Returns a mesh axis that does not exist, e.g. legacy's name for `tp`."""

  def map_axis(self, axis, tensor_name, tensor_type):
    return "tensor"


class ResolveAxisSizesTest(parameterized.TestCase):
  """The mesh arithmetic, tested without needing devices.

  The same function resolves both shapes; only the target it divides into
  changes. So the ICI and DCN cases below are the same code exercised against
  a device count and a slice count respectively.
  """

  @parameterized.named_parameters(
      ("all_explicit", (1, 4, 2, 1), 8, (1, 4, 2, 1)),
      ("auto_fsdp", (1, -1, 1, 1), 8, (1, 8, 1, 1)),
      ("auto_fsdp_with_tp", (1, -1, 2, 1), 8, (1, 4, 2, 1)),
      ("auto_dp", (-1, 2, 2, 1), 8, (2, 2, 2, 1)),
      ("auto_expert", (1, 2, 1, -1), 8, (1, 2, 1, 4)),
      ("single_device", (1, -1, 1, 1), 1, (1, 1, 1, 1)),
  )
  def test_resolves(self, sizes, target, expected):
    self.assertEqual(_resolve_axis_sizes(sizes, target, "ICI", "per-slice device count"), expected)

  @parameterized.named_parameters(
      ("auto_dp_over_slices", (-1, 1, 1, 1), 4, (4, 1, 1, 1)),
      ("explicit_split", (2, 2, 1, 1), 4, (2, 2, 1, 1)),
      ("one_slice", (-1, 1, 1, 1), 1, (1, 1, 1, 1)),
  )
  def test_resolves_dcn_against_slice_count(self, sizes, num_slices, expected):
    self.assertEqual(_resolve_axis_sizes(sizes, num_slices, "DCN", "slice count"), expected)

  def test_rejects_multiple_auto_axes(self):
    with self.assertRaisesRegex(ValueError, "At most one ICI mesh axis may be -1"):
      _resolve_axis_sizes((-1, -1, 1, 1), 8, "ICI", "per-slice device count")

  @parameterized.named_parameters(
      ("zero", (0, 1, 1, 1)),
      ("negative", (1, -2, 1, 1)),
  )
  def test_rejects_invalid_sizes(self, sizes):
    with self.assertRaisesRegex(ValueError, "must be >= 1"):
      _resolve_axis_sizes(sizes, 8, "ICI", "per-slice device count")

  def test_rejects_indivisible_auto_axis(self):
    """8 devices cannot be split with tp=3."""
    with self.assertRaisesRegex(ValueError, "not divisible"):
      _resolve_axis_sizes((1, -1, 3, 1), 8, "ICI", "per-slice device count")

  @parameterized.named_parameters(
      ("too_few", (1, 2, 1, 1), 8),
      ("too_many", (2, 8, 1, 1), 8),
  )
  def test_rejects_product_mismatch(self, sizes, target):
    with self.assertRaisesRegex(ValueError, "does not match"):
      _resolve_axis_sizes(sizes, target, "ICI", "per-slice device count")

  def test_errors_name_the_network_they_came_from(self):
    """An ICI misconfiguration and a DCN one need different fixes.

    A message saying only "mesh axes" would leave the reader guessing which of
    the two shapes they got wrong, and the counts involved differ by orders of
    magnitude.
    """
    with self.assertRaisesRegex(ValueError, r"ICI mesh axes .* per-slice device count \(8\)"):
      _resolve_axis_sizes((1, 2, 1, 1), 8, "ICI", "per-slice device count")
    with self.assertRaisesRegex(ValueError, r"DCN mesh axes .* slice count \(2\)"):
      _resolve_axis_sizes((1, 1, 1, 1), 2, "DCN", "slice count")


class CountSlicesTest(parameterized.TestCase):
  """Slice detection.

  Stand-in objects rather than real devices: slice membership is just an
  attribute, and a single-slice testbed cannot produce a device carrying a
  non-zero one.
  """

  def test_devices_without_the_attribute_count_as_one_slice(self):
    """The `getattr` default is load bearing, not defensive.

    Verified on both testbeds used here: CPU devices and single-slice v6e
    devices have no `slice_index` attribute at all, so reading it directly
    would raise on every single-slice run.
    """
    self.assertFalse(hasattr(jax.devices()[0], "slice_index"))
    self.assertEqual(_count_slices(jax.devices()), 1)

  def test_counts_distinct_indices(self):
    devices = [SimpleNamespace(slice_index=i // 4) for i in range(8)]
    self.assertEqual(_count_slices(devices), 2)

  def test_uniform_index_is_one_slice(self):
    devices = [SimpleNamespace(slice_index=0) for _ in range(8)]
    self.assertEqual(_count_slices(devices), 1)


class ShardingRulesTest(parameterized.TestCase):
  """Logical axis -> `PartitionSpec` translation."""

  def setUp(self):
    super().setUp()
    self.sharding = _ExampleSharding()

  def test_weight_spec(self):
    spec = self.sharding("q_proj", ("embed", "heads", "head_dim"))
    self.assertEqual(spec, P("fsdp", "tp", None))

  def test_defaults_to_weight_tensor_type(self):
    """Omitting `tensor_type` must not silently produce activation rules."""
    self.assertEqual(
        self.sharding("tok_embed", ("embed",)),
        self.sharding("tok_embed", ("embed",), TensorType.WEIGHT),
    )

  def test_activation_spec_differs_from_weight(self):
    """`embed` shards weights over FSDP but leaves activations replicated."""
    self.assertEqual(self.sharding("x", ("embed",), TensorType.WEIGHT), P("fsdp"))
    self.assertEqual(self.sharding("x", ("embed",), TensorType.ACTIVATION), P(None))

  def test_multi_axis_mapping(self):
    """A logical axis may consume several mesh axes."""
    spec = self.sharding("x", ("batch", "length", "embed"), TensorType.ACTIVATION)
    self.assertEqual(spec, P(("dp", "fsdp"), None, None))

  def test_none_axis_is_replicated_without_consulting_rules(self):
    """An unnamed dimension bypasses `map_axis` entirely."""
    spec = self.sharding("scale", (None, "mlp"))
    self.assertEqual(spec, P(None, "tp"))

  def test_unmapped_axis_raises(self):
    """An unrecognized logical axis must fail loudly, not replicate silently.

    `embedding` is the plausible error: the full word where the rule set spells
    the axis `embed`.
    """
    with self.assertRaises(KeyError):
      self.sharding("q_proj", ("embedding",))

  def test_unknown_mesh_axis_raises(self):
    with self.assertRaisesRegex(ValueError, "unknown mesh axes"):
      _BadSharding()("q_proj", ("embed",))

  def test_empty_axes_gives_fully_replicated_spec(self):
    self.assertEqual(self.sharding("scalar", ()), P())


class _ConflictingSharding(Sharding):
  """Sends two different logical axes to the same mesh axis."""

  def map_axis(self, axis, tensor_name, tensor_type):
    return "fsdp" if axis in ("embed", "mlp") else None


class MeshAxisConflictTest(parameterized.TestCase):
  """A mesh axis may shard only one dimension of a given tensor.

  Legacy tolerates rules that would collide because Flax skips a rule whose
  mesh axis is already claimed by another dimension of the same array and
  falls through to the next matching rule -- which is why `logical_axis_rules`
  contains duplicate keys such as two `embed` entries of differing length.

  m3 has no fallback: `map_axis` returns one answer. So the collision is
  detected and reported against the rule that caused it, rather than reaching
  XLA as an invalid `PartitionSpec`.
  """

  def test_two_dimensions_claiming_one_mesh_axis_raises(self):
    with self.assertRaisesRegex(ValueError, "mesh axis 'fsdp'"):
      _ConflictingSharding()("w", ("embed", "mlp"))

  def test_error_names_both_logical_axes(self):
    """The message has to identify the pair, or the rule set is a haystack."""
    with self.assertRaises(ValueError) as ctx:
      _ConflictingSharding()("w", ("embed", "mlp"))
    self.assertIn("'embed'", str(ctx.exception))
    self.assertIn("'mlp'", str(ctx.exception))

  def test_conflict_within_a_multi_axis_mapping_is_caught(self):
    """`("dp", "fsdp")` on one dimension still consumes both axes."""

    class _Overlapping(Sharding):

      def map_axis(self, axis, tensor_name, tensor_type):
        return ("dp", "fsdp") if axis == "batch" else "fsdp"

    with self.assertRaisesRegex(ValueError, "mesh axis 'fsdp'"):
      _Overlapping()("x", ("batch", "embed"))

  def test_repeated_none_is_not_a_conflict(self):
    """Replicated dimensions consume nothing, so any number of them is fine."""
    self.assertEqual(_ConflictingSharding()("w", ("a", "b", "c")), P(None, None, None))


class _RecordingSharding(_ExampleSharding):
  """Records every `map_axis` call so the `tensor_type` argument is observable."""

  def __init__(self):
    self.calls = []

  def map_axis(self, axis, tensor_name, tensor_type):
    self.calls.append((axis, tensor_name, tensor_type))
    return super().map_axis(axis, tensor_name, tensor_type)


class ConstrainTest(parameterized.TestCase):
  """`constrain` is the activation-only entry point.

  These assert the contract `constrain` implements, not the sharding XLA ends
  up realizing. The realized spec is normalized -- size-1 mesh axes are dropped
  and a single-device constraint is erased entirely -- so asserting on it makes
  a test of our rules into a test of XLA's device count.
  """

  def test_passes_activation_tensor_type(self):
    """`constrain` must request activation rules, never weight rules.

    This is load bearing: `embed` maps to `fsdp` for weights and to None for
    activations, so silently using weight rules here would shard activations
    that are meant to stay replicated.
    """
    sharding = _RecordingSharding()
    mesh = create_mesh(devices=jax.devices()[:1])

    with jax.sharding.set_mesh(mesh):
      jax.jit(lambda x: sharding.constrain(x, "hidden", ("batch", "embed")))(jnp.zeros((1, 4)))

    self.assertNotEmpty(sharding.calls)
    for axis, _, tensor_type in sharding.calls:
      self.assertEqual(tensor_type, TensorType.ACTIVATION, f"axis {axis!r} was resolved with the wrong tensor type")

  def test_preserves_values(self):
    """A sharding constraint is a placement hint; it must not alter the data."""
    sharding = _ExampleSharding()
    mesh = create_mesh(devices=jax.devices()[:1])
    x = jnp.arange(8.0).reshape(1, 8)

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda v: sharding.constrain(v, "hidden", ("batch", "embed")))(x)

    self.assertTrue(bool(jnp.array_equal(out, x)))

  def test_single_device_constraint_is_erased(self):
    """Documents why the single-device path needs no special casing.

    With one device every axis has size 1, so XLA normalizes the constraint
    away to a fully replicated `P()`. Model code can therefore call `constrain`
    unconditionally.
    """
    sharding = _ExampleSharding()
    mesh = create_mesh(devices=jax.devices()[:1])

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "hidden", ("batch", "embed")))(jnp.zeros((1, 4)))

    self.assertEqual(out.sharding.spec, P())


def _as_explicit(mesh: jax.sharding.Mesh) -> jax.sharding.Mesh:
  """Relabels a mesh's axes as `Explicit`, keeping its device placement.

  `create_mesh` only builds auto meshes today, so the explicit-mode tests
  construct theirs here rather than through the public API.
  """
  return jax.sharding.Mesh(
      mesh.devices, MESH_AXIS_NAMES, axis_types=(jax.sharding.AxisType.Explicit,) * len(MESH_AXIS_NAMES)
  )


class ShardModeTest(parameterized.TestCase):
  """`constrain` must pick the API that matches the mesh's sharding mode.

  The two are not interchangeable: under an explicit mesh
  `with_sharding_constraint` asserts the array is *already* sharded as asked,
  and `reshard` rejects a spec naming auto axes. Calling the wrong one raises.
  """

  def test_auto_mesh_is_not_explicit(self):
    mesh = create_mesh(devices=jax.devices()[:1])
    with jax.sharding.set_mesh(mesh):
      self.assertFalse(_ambient_mesh_is_explicit())

  def test_explicit_mesh_is_detected(self):
    mesh = _as_explicit(create_mesh(devices=jax.devices()[:1]))
    with jax.sharding.set_mesh(mesh):
      self.assertTrue(_ambient_mesh_is_explicit())

  def test_mixed_axis_types_are_rejected(self):
    """No single API is right for a spec spanning both kinds of axis."""
    base = create_mesh(devices=jax.devices()[:1])
    mixed = jax.sharding.Mesh(
        base.devices,
        MESH_AXIS_NAMES,
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto) * 2,
    )
    with jax.sharding.set_mesh(mixed):
      with self.assertRaisesRegex(ValueError, "mixes explicit and non-explicit"):
        _ambient_mesh_is_explicit()

  def test_constrain_works_under_an_explicit_mesh(self):
    """Regression test: this raised `AssertionError` before the dispatch existed."""
    sharding = _ExampleSharding()
    mesh = _as_explicit(create_mesh(devices=jax.devices()[:1]))
    x = jnp.arange(8.0).reshape(1, 8)

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda v: sharding.constrain(v, "hidden", ("batch", "embed")))(x)

    self.assertTrue(bool(jnp.array_equal(out, x)))

  def test_constrain_still_requests_activation_rules_when_explicit(self):
    """The tensor-type contract must not depend on which API is dispatched to."""
    sharding = _RecordingSharding()
    mesh = _as_explicit(create_mesh(devices=jax.devices()[:1]))

    with jax.sharding.set_mesh(mesh):
      jax.jit(lambda x: sharding.constrain(x, "hidden", ("batch", "embed")))(jnp.zeros((1, 4)))

    self.assertNotEmpty(sharding.calls)
    for axis, _, tensor_type in sharding.calls:
      self.assertEqual(tensor_type, TensorType.ACTIVATION, f"axis {axis!r} was resolved with the wrong tensor type")


class CreateMeshTest(parameterized.TestCase):
  """Mesh construction. Single-device cases run anywhere."""

  def test_axis_names_are_stable(self):
    """Model rules reference these by name; reordering silently rewires sharding."""
    self.assertEqual(MESH_AXIS_NAMES, ("dp", "fsdp", "tp", "ep"))

  def test_single_device_mesh_is_trivial(self):
    """Defaults must work on one device so there is no special-case code path."""
    mesh = create_mesh(devices=jax.devices()[:1])
    self.assertEqual(mesh.axis_names, MESH_AXIS_NAMES)
    self.assertEqual(mesh.shape["dp"], 1)
    self.assertEqual(mesh.shape["fsdp"], 1)
    self.assertEqual(mesh.devices.size, 1)

  def test_rejects_bad_axis_sizes(self):
    with self.assertRaises(ValueError):
      create_mesh(ici={"dp": -1, "fsdp": -1}, devices=jax.devices()[:1])

  def test_dcn_defaults_are_inert_on_one_slice(self):
    """`dcn_dp_parallelism=-1` must resolve to 1, not compete with the ICI shape.

    The DCN defaults are always applied, including on the single-slice runs
    that are by far the most common, so they have to disappear cleanly there.
    """
    mesh = create_mesh(devices=jax.devices()[:1])
    self.assertEqual(mesh.devices.size, 1)
    self.assertEqual(mesh.shape["dp"], 1)

  def test_ici_and_dcn_may_each_be_auto(self):
    """One `-1` per shape is legal, because they resolve against different totals."""
    mesh = create_mesh(ici={"fsdp": -1}, dcn={"dp": -1}, devices=jax.devices()[:1])
    self.assertEqual(mesh.devices.size, 1)

  @parameterized.named_parameters(
      # An explicit `-1` with nothing left for it to absorb: fails auto-sizing.
      ("with_auto_dp", {"dcn": {"dp": -1, "tp": 2}}, "Cannot auto-size the -1 DCN mesh axis"),
      # Fully specified, so it reaches the product check instead.
      ("fully_specified", {"dcn": {"dp": 1, "tp": 2}}, r"DCN mesh axes .* slice count \(1\)"),
  )
  def test_rejects_dcn_request_that_exceeds_the_slice_count(self, kwargs, message):
    """Asking for cross-slice parallelism on a one-slice job must fail loudly.

    Quietly ignoring it would run a single-slice job that looks like the
    multi-slice one the user asked for. Which of the two errors fires depends
    on whether a `-1` is present, so both paths are pinned here.
    """
    with self.assertRaisesRegex(ValueError, message):
      create_mesh(devices=jax.devices()[:1], **kwargs)

  def test_rejects_multiple_dcn_auto_axes(self):
    with self.assertRaisesRegex(ValueError, "At most one DCN mesh axis"):
      create_mesh(dcn={"dp": -1, "fsdp": -1}, devices=jax.devices()[:1])

  def test_unnamed_axes_default_to_one(self):
    """Naming only what you split is the point of the mapping form.

    Note this differs from a per-axis-keyword API, where an omitted argument
    keeps that axis's own default. Here an omitted axis is simply 1.
    """
    mesh = create_mesh(ici={"fsdp": -1}, devices=jax.devices()[:1])
    for name in MESH_AXIS_NAMES:
      self.assertEqual(mesh.shape[name], 1)

  @parameterized.named_parameters(
      ("ici", {"ici": {"fsdpp": -1}}, "Unknown ICI mesh axes"),
      ("dcn", {"dcn": {"pipeline": 2}}, "Unknown DCN mesh axes"),
  )
  def test_rejects_unknown_axis_names(self, kwargs, message):
    """A typo must not silently leave the axis at 1 and shard nothing."""
    with self.assertRaisesRegex(ValueError, message):
      create_mesh(devices=jax.devices()[:1], **kwargs)

  def test_empty_mapping_is_all_replicated(self):
    mesh = create_mesh(ici={}, dcn={}, devices=jax.devices()[:1])
    self.assertEqual(mesh.devices.size, 1)

  def test_rejects_empty_devices(self):
    """An empty mapping is legal; an empty device list is not.

    Without this guard `_count_slices` returns 0 and the divisibility check
    below it raises `ZeroDivisionError`, which says nothing about the cause.
    """
    with self.assertRaisesRegex(ValueError, "zero devices"):
      create_mesh(devices=[])


@pytest.mark.tpu_only
class CreateMeshMultiDeviceTest(parameterized.TestCase):
  """Cases that need a real multi-device mesh.

  Marked `tpu_only` deliberately: unmarked tests are auto-marked `cpu_only` by
  the repo conftest and skipped on accelerator testbeds, which is exactly where
  these need to run.
  """

  def setUp(self):
    super().setUp()
    self.num_devices = jax.device_count()
    if self.num_devices < 8:
      self.skipTest(f"needs at least 8 devices, found {self.num_devices}")

  def test_auto_fsdp_absorbs_all_devices(self):
    mesh = create_mesh(devices=jax.devices()[:8])
    self.assertEqual(mesh.shape["fsdp"], 8)
    self.assertEqual(mesh.shape["tp"], 1)

  def test_tensor_parallelism_splits_with_fsdp(self):
    mesh = create_mesh(ici={"fsdp": -1, "tp": 2}, devices=jax.devices()[:8])
    self.assertEqual(mesh.shape["fsdp"], 4)
    self.assertEqual(mesh.shape["tp"], 2)

  def test_constrain_shards_activations_across_mesh(self):
    """End to end: logical axes actually split data across devices.

    Asserted via per-device shard shape rather than the `PartitionSpec`. With
    tp=2 the mesh is dp=1, fsdp=4, tp=2, and XLA drops the size-1 `dp` from the
    realized spec -- so `("batch", ...)` comes back as `P("fsdp", ...)` rather
    than `P(("dp", "fsdp"), ...)`. Shard shape states the property we actually
    care about and does not move when the mesh does.
    """
    sharding = _ExampleSharding()
    mesh = create_mesh(ici={"fsdp": -1, "tp": 2}, devices=jax.devices()[:8])

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "mlp_mid", ("batch", "length", "mlp")))(jnp.zeros((4, 2, 16)))

    # batch 4 over fsdp=4, length replicated, mlp 16 over tp=2.
    self.assertEqual(out.sharding.shard_shape(out.shape), (1, 2, 8))

  def test_replicated_axes_are_not_sharded(self):
    """A rule returning None must leave the dimension whole on every device."""
    sharding = _ExampleSharding()
    mesh = create_mesh(ici={"fsdp": -1, "tp": 2}, devices=jax.devices()[:8])

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "hidden", ("batch", "length", "embed")))(jnp.zeros((4, 2, 16)))

    # `embed` is unconstrained for activations, so the last dim stays at 16.
    self.assertEqual(out.sharding.shard_shape(out.shape), (1, 2, 16))

  def test_constrain_shards_activations_under_an_explicit_mesh(self):
    """The explicit path must shard the same way the auto path does.

    This is the case that actually caught the bug. On a single device every
    axis has size 1, so `with_sharding_constraint`'s assertion passes and the
    wrong API goes unnoticed; with real shards it raises.

    The realized spec also differs between modes -- explicit meshes keep the
    size-1 `dp` that auto normalizes away -- so this asserts shard shape, which
    is the property that has to match.
    """
    sharding = _ExampleSharding()
    mesh = _as_explicit(create_mesh(ici={"fsdp": -1, "tp": 2}, devices=jax.devices()[:8]))

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "mlp_mid", ("batch", "length", "mlp")))(jnp.zeros((4, 2, 16)))

    self.assertEqual(out.sharding.shard_shape(out.shape), (1, 2, 8))


@pytest.mark.tpu_only
class CreateMeshMultiSliceTest(parameterized.TestCase):
  """The DCN branch, with the pieces m3 does not own stubbed out.

  Two things are faked. The slice count, because a single-slice testbed cannot
  produce devices carrying a `slice_index`. And jax's placement algorithm,
  replaced by a plain reshape, because `create_hybrid_device_mesh` reads that
  same attribute off every device. Neither is m3's code.

  What is left is what m3 owns and could get wrong: that the ICI shape is
  resolved against devices-per-slice rather than the total device count, that
  the DCN shape is resolved against the slice count, and that both reach the
  hybrid builder.
  """

  def setUp(self):
    super().setUp()
    if jax.device_count() < 8:
      self.skipTest(f"needs at least 8 devices, found {jax.device_count()}")

  def test_ici_resolves_per_slice_and_dcn_across_slices(self):
    recorded = {}

    def fake_hybrid(ici_shape, dcn_shape, devices, **_):
      recorded["ici"] = tuple(ici_shape)
      recorded["dcn"] = tuple(dcn_shape)
      return np.array(devices).reshape(np.multiply(ici_shape, dcn_shape))

    with mock.patch.object(sharding_lib, "_count_slices", return_value=2):
      with mock.patch.object(sharding_lib.mesh_utils, "create_hybrid_device_mesh", side_effect=fake_hybrid):
        mesh = create_mesh(devices=jax.devices()[:8])

    # 8 devices across 2 slices is 4 per slice. The default `ici_fsdp=-1` must
    # absorb 4, not 8 -- absorbing 8 is the bug this test exists to catch.
    self.assertEqual(recorded["ici"], (1, 4, 1, 1))
    self.assertEqual(recorded["dcn"], (2, 1, 1, 1))
    self.assertEqual(mesh.shape["dp"], 2)
    self.assertEqual(mesh.shape["fsdp"], 4)

  def test_single_slice_does_not_take_the_hybrid_path(self):
    """`create_hybrid_device_mesh` raises on devices without a `slice_index`.

    It does so even when the DCN shape is all ones, so the branch cannot be
    collapsed into an unconditional hybrid call. This pins that.
    """
    with mock.patch.object(sharding_lib.mesh_utils, "create_hybrid_device_mesh") as hybrid:
      create_mesh(devices=jax.devices()[:8])
    hybrid.assert_not_called()

  def test_dcn_shape_must_match_the_slice_count(self):
    with mock.patch.object(sharding_lib, "_count_slices", return_value=2):
      with self.assertRaisesRegex(ValueError, r"DCN mesh axes .* slice count \(2\)"):
        create_mesh(dcn={"dp": 4}, devices=jax.devices()[:8])

  def test_rejects_devices_that_do_not_divide_into_slices(self):
    """Integer division would silently drop devices; that must raise instead."""
    with mock.patch.object(sharding_lib, "_count_slices", return_value=3):
      with self.assertRaisesRegex(ValueError, "do not divide evenly"):
        create_mesh(devices=jax.devices()[:8])


if __name__ == "__main__":
  absltest.main()

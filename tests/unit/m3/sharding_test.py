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
without it would silently never run anywhere it matters. Those tests size
their meshes and expectations from `jax.device_count()` rather than assuming
eight: the TPU runner CI uses has four chips, and a hard-coded eight turned
every one of them into a skip.

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

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.m3.core import sharding as sharding_lib
from maxtext.m3.core.sharding import DefaultSharding
from maxtext.m3.core.sharding import MESH_AXIS_NAMES
from maxtext.m3.core.sharding import Sharding
from maxtext.m3.core.sharding import TensorType
from maxtext.m3.core.sharding import _ambient_mesh_is_explicit
from maxtext.m3.core.sharding import _count_slices
from maxtext.m3.core.sharding import _resolve_axis_sizes
from maxtext.m3.core.sharding import create_mesh


class _ExampleSharding(Sharding):
  """A small rule set for exercising the ABC, independent of `DefaultSharding`.

  Kept separate so the mechanics tests below keep passing if the default rules
  change; `DefaultShardingTest` pins those rules on their own.
  """

  def map_axis(self, axis, tensor_name, tensor_type):
    match axis, tensor_type:
      case "batch", TensorType.ACTIVATION:
        return ("data", "fsdp")
      case "embed", TensorType.WEIGHT:
        return "fsdp"
      case "embed", TensorType.ACTIVATION:
        return None
      case ("heads" | "kv_heads"), _:
        return "tensor"
      case "head_dim", _:
        return None
      case "mlp", _:
        return "tensor"
      case "length", _:
        return None
      case _:
        raise KeyError(f"unmapped logical axis {axis!r} on tensor {tensor_name!r}")


class _BadSharding(Sharding):
  """Returns a mesh axis that does not exist: the abbreviation for `tensor`."""

  def map_axis(self, axis, tensor_name, tensor_type):
    return "tp"


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
      ("auto_data", (-1, 2, 2, 1), 8, (2, 2, 2, 1)),
      ("auto_expert", (1, 2, 1, -1), 8, (1, 2, 1, 4)),
      ("single_device", (1, -1, 1, 1), 1, (1, 1, 1, 1)),
  )
  def test_resolves(self, sizes, target, expected):
    self.assertEqual(_resolve_axis_sizes(sizes, target, "ICI", "per-slice device count"), expected)

  @parameterized.named_parameters(
      ("auto_data_over_slices", (-1, 1, 1, 1), 4, (4, 1, 1, 1)),
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
    """8 devices cannot be split with tensor=3."""
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
    self.assertEqual(spec, P("fsdp", "tensor", None))

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
    self.assertEqual(spec, P(("data", "fsdp"), None, None))

  def test_none_axis_is_replicated_without_consulting_rules(self):
    """An unnamed dimension bypasses `map_axis` entirely."""
    spec = self.sharding("scale", (None, "mlp"))
    self.assertEqual(spec, P(None, "tensor"))

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
    """`("data", "fsdp")` on one dimension still consumes both axes."""

    class _Overlapping(Sharding):

      def map_axis(self, axis, tensor_name, tensor_type):
        return ("data", "fsdp") if axis == "batch" else "fsdp"

    with self.assertRaisesRegex(ValueError, "mesh axis 'fsdp'"):
      _Overlapping()("x", ("batch", "embed"))

  def test_repeated_none_is_not_a_conflict(self):
    """Replicated dimensions consume nothing, so any number of them is fine."""
    self.assertEqual(_ConflictingSharding()("w", ("a", "b", "c")), P(None, None, None))


class _RecordingSharding(_ExampleSharding):
  """Records every `map_axis` call so the `tensor_type` argument is observable."""

  def __init__(self):
    super().__init__()
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


def _require_multi_device(test: absltest.TestCase) -> int:
  """Skips `test` unless the testbed has an even device count of at least four; returns the count.

  Even, because the meshes below split `tensor=2`; at least four, because the
  expert-axis case needs `fsdp=2` times `expert>=2`. That admits the four-chip
  TPU runner CI uses, eight-chip hosts, and eight forced host-CPU devices alike.
  """
  n = jax.device_count()
  if n < 4 or n % 2:
    test.skipTest(f"needs an even device count of at least 4, found {n}")
  return n


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


class DefaultShardingTest(parameterized.TestCase):
  """`DefaultSharding` pinned against the tensors in the design doc's Llama.

  Exact specs rather than "it resolved": a rule that lands on the wrong axis
  still trains, just slower, and nothing else would notice.
  """

  def setUp(self):
    super().setUp()
    # Flax eager sharding is a process-global flag, and other code sharing the
    # test process can turn it off. Without it, `*_metadata` sharding is neither
    # enforced nor applied.
    self.enter_context(nnx.use_eager_sharding(True))
    self.sharding = DefaultSharding()

  @parameterized.named_parameters(
      ("q_proj", "q_proj", ("embed", "heads", "head_dim"), P(("fsdp", "expert"), "tensor", None)),
      ("k_proj", "k_proj", ("embed", "kv_heads", "head_dim"), P(("fsdp", "expert"), "tensor", None)),
      ("o_proj", "o_proj", ("heads", "head_dim", "embed"), P("tensor", None, ("fsdp", "expert"))),
      ("gate", "gate", ("embed", "mlp"), P(("fsdp", "expert"), "tensor")),
      ("down", "down", ("mlp", "embed"), P("tensor", ("fsdp", "expert"))),
      ("embedding", "embed", ("vocab", "embed"), P("tensor", ("fsdp", "expert"))),
      ("norm_scale", "attn_norm", ("norm",), P(None)),
  )
  def test_weight_specs(self, tensor_name, axes, expected):
    self.assertEqual(self.sharding(tensor_name, axes), expected)

  @parameterized.named_parameters(
      ("query", "query", ("batch", "length", "heads", "head_dim"), P(("data", "fsdp", "expert"), None, "tensor", None)),
      ("key", "key", ("batch", "length", "kv_heads", "head_dim"), P(("data", "fsdp", "expert"), None, "tensor", None)),
      ("residual", "post_attn", ("batch", "length", "embed"), P(("data", "fsdp", "expert"), None, None)),
      ("mlp_mid", "mlp_mid", ("batch", "length", "mlp"), P(("data", "fsdp", "expert"), None, "tensor")),
      ("logits", "logits", ("batch", "length", "vocab"), P(("data", "fsdp", "expert"), None, "tensor")),
  )
  def test_activation_specs(self, tensor_name, axes, expected):
    self.assertEqual(self.sharding(tensor_name, axes, TensorType.ACTIVATION), expected)

  @parameterized.parameters("embed", "heads", "kv_heads", "head_dim", "mlp", "vocab", "norm", "length")
  def test_logical_axes_resolve_for_both_tensor_types(self, axis):
    """Each logical axis a model may pass has a rule for both tensor types.

    `batch` is the deliberate exception -- activation-only, pinned by
    `test_batch_is_not_a_weight_axis` below.
    """
    for tensor_type in TensorType:
      self.sharding("t", (axis,), tensor_type)

  def test_unknown_axis_raises(self):
    with self.assertRaisesRegex(KeyError, "embedding"):
      self.sharding("tok_embed", ("embedding", "embed"))

  def test_batch_is_not_a_weight_axis(self):
    """A weight annotated with `batch` is a model bug, not a replicated dimension."""
    with self.assertRaises(KeyError):
      self.sharding("w", ("batch", "embed"))

  def test_no_doc_tensor_conflicts(self):
    """The conflict check in `__call__` must stay silent for every tensor the doc's Llama declares."""
    weights = [("embed", "heads", "head_dim"), ("heads", "head_dim", "embed"), ("embed", "mlp"), ("vocab", "embed")]
    activations = [("batch", "length", "heads", "head_dim"), ("batch", "length", "embed"), ("batch", "length", "vocab")]
    for axes in weights:
      self.sharding("w", axes)
    for axes in activations:
      self.sharding("a", axes, TensorType.ACTIVATION)

  def test_construction_outside_a_mesh_raises(self):
    """Pins the protocol: construct under `set_mesh`, even on one device.

    Flax resolves `kernel_metadata={"out_sharding": ...}` eagerly against the mesh
    in scope and raises when there is none, rather than storing the spec and
    leaving the parameter replicated. So the single-device path is "a trivial
    mesh", never "no mesh". `set_mesh(None)` clears any mesh another test may
    have left in scope; eager sharding is pinned on in `setUp`.
    """
    spec = self.sharding("gate", ("embed", "mlp"))
    with jax.sharding.set_mesh(None):
      with self.assertRaisesRegex(ValueError, "mesh"):
        nnx.Linear(4, 8, use_bias=False, kernel_metadata={"out_sharding": spec}, rngs=nnx.Rngs(0))

  def test_construction_under_a_trivial_mesh_succeeds(self):
    """The other half of the protocol: a one-device mesh is enough."""
    spec = self.sharding("gate", ("embed", "mlp"))
    with jax.sharding.set_mesh(create_mesh(devices=jax.devices()[:1])):
      layer = nnx.Linear(4, 8, use_bias=False, kernel_metadata={"out_sharding": spec}, rngs=nnx.Rngs(0))
    self.assertEqual(layer.kernel[...].shape, (4, 8))


@pytest.mark.tpu_only
class DefaultShardingMultiDeviceTest(parameterized.TestCase):
  """`DefaultSharding` driving real parameter placement through NNX metadata.

  Marked `tpu_only` for the same reason as `CreateMeshMultiDeviceTest`. Sizes
  come from the device count `n`: `tensor=2` leaves `fsdp = n/2`, and the
  expert case uses `fsdp=2, expert = n/2`. Tensor dimensions are chosen as
  multiples of those so every expected shard shape is an exact integer.
  """

  def setUp(self):
    super().setUp()
    self.n = _require_multi_device(self)
    self.enter_context(nnx.use_eager_sharding(True))  # See `DefaultShardingTest.setUp`.
    self.sharding = DefaultSharding()
    self.devices = jax.devices()

  def test_parameters_are_placed_at_construction(self):
    """`*_metadata={"out_sharding": ...}` plus an ambient mesh is the whole parameter-sharding mechanism.

    Asserted via per-device shard shape, which does not depend on how the
    realized spec is normalized.
    """
    s = self.sharding
    fsdp = self.n // 2
    embed_dim = 4 * fsdp
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)

    with jax.sharding.set_mesh(mesh):
      gate = nnx.Linear(
          embed_dim, 32, use_bias=False, kernel_metadata={"out_sharding": s("gate", ("embed", "mlp"))}, rngs=nnx.Rngs(0)
      )
      embed = nnx.Embed(
          64, embed_dim, embedding_metadata={"out_sharding": s("embed", ("vocab", "embed"))}, rngs=nnx.Rngs(0)
      )
      norm = nnx.RMSNorm(embed_dim, scale_metadata={"out_sharding": s("norm", ("norm",))}, rngs=nnx.Rngs(0))

    # embed over fsdp (4 per device), mlp 32 over tensor=2.
    self.assertEqual(gate.kernel[...].sharding.shard_shape((embed_dim, 32)), (4, 16))
    # vocab 64 over tensor=2, embed over fsdp.
    self.assertEqual(embed.embedding[...].sharding.shard_shape((64, embed_dim)), (32, 4))
    # norm scale replicated.
    self.assertEqual(norm.scale[...].sharding.shard_shape((embed_dim,)), (embed_dim,))

  def test_mesh_shape_decides_which_axes_are_live(self):
    """Same rules, two meshes: the strategy does not change, the combination does.

    With `tensor=1` the `tensor` entries shard nothing and the spec is
    equivalent to pure FSDP; with `tensor=2` it is not. This is the property
    that lets one rule set serve every combination of the axes it names.
    """
    spec = self.sharding("gate", ("embed", "mlp"))  # P(("fsdp", "expert"), "tensor")
    pure_fsdp = P("fsdp", None)

    mesh = create_mesh(ici={"fsdp": -1}, devices=self.devices)
    self.assertTrue(NamedSharding(mesh, spec).is_equivalent_to(NamedSharding(mesh, pure_fsdp), 2))

    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    self.assertFalse(NamedSharding(mesh, spec).is_equivalent_to(NamedSharding(mesh, pure_fsdp), 2))

  def test_expert_axis_shards_dense_tensors(self):
    """On a mesh that splits `expert`, dense weights and the batch are sharded across it.

    This is why `expert` sits in the `batch` and `embed` unions. With `embed`
    on `fsdp` alone, a dense weight would be replicated once per `expert`
    shard -- `expert` times the memory -- and the batch would not be split over
    those devices at all. The single-axis spec is evaluated alongside to show
    the difference rather than assert it from memory.
    """
    s = self.sharding
    n = self.n
    embed_dim = 2 * n
    mesh = create_mesh(ici={"fsdp": 2, "expert": -1}, devices=self.devices)
    self.assertEqual(mesh.shape["expert"], n // 2)

    with jax.sharding.set_mesh(mesh):
      gate = nnx.Linear(
          embed_dim, 32, use_bias=False, kernel_metadata={"out_sharding": s("gate", ("embed", "mlp"))}, rngs=nnx.Rngs(0)
      )
      residual = jax.jit(lambda x: s.constrain(x, "post_attn", ("batch", "length", "embed")))(
          jnp.zeros((n, 2, embed_dim))
      )

    # embed over fsdp x expert = n devices, 2 rows each; mlp 32 over tensor = 1.
    self.assertEqual(gate.kernel[...].sharding.shard_shape((embed_dim, 32)), (2, 32))
    # batch n over data x fsdp x expert = n.
    self.assertEqual(residual.sharding.shard_shape(residual.shape), (1, 2, embed_dim))
    # The rule this replaces: `fsdp` alone leaves each device holding `expert` times as much.
    self.assertEqual(NamedSharding(mesh, P("fsdp", None)).shard_shape((embed_dim, 32)), (n, 32))

  def test_activation_constraints_apply(self):
    s = self.sharding
    fsdp = self.n // 2
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)

    with jax.sharding.set_mesh(mesh):
      logits = jax.jit(lambda x: s.constrain(x, "logits", ("batch", "length", "vocab")))(jnp.zeros((fsdp, 2, 16)))
      residual = jax.jit(lambda x: s.constrain(x, "post_attn", ("batch", "length", "embed")))(jnp.zeros((fsdp, 2, 16)))

    # batch over fsdp; vocab 16 over tensor=2; embed replicated.
    self.assertEqual(logits.sharding.shard_shape(logits.shape), (1, 2, 8))
    self.assertEqual(residual.sharding.shard_shape(residual.shape), (1, 2, 16))


class CreateMeshTest(parameterized.TestCase):
  """Mesh construction. Single-device cases run anywhere."""

  def test_axis_names_are_stable(self):
    """Model rules reference these by name; reordering silently rewires sharding."""
    self.assertEqual(MESH_AXIS_NAMES, ("data", "fsdp", "tensor", "expert"))

  def test_single_device_mesh_is_trivial(self):
    """Defaults must work on one device so there is no special-case code path."""
    mesh = create_mesh(devices=jax.devices()[:1])
    self.assertEqual(mesh.axis_names, MESH_AXIS_NAMES)
    self.assertEqual(mesh.shape["data"], 1)
    self.assertEqual(mesh.shape["fsdp"], 1)
    self.assertEqual(mesh.devices.size, 1)

  def test_rejects_bad_axis_sizes(self):
    with self.assertRaises(ValueError):
      create_mesh(ici={"data": -1, "fsdp": -1}, devices=jax.devices()[:1])

  def test_dcn_defaults_are_inert_on_one_slice(self):
    """`dcn={"data": -1}` must resolve to 1, not compete with the ICI shape.

    The DCN defaults are always applied, including on the single-slice runs
    that are by far the most common, so they have to disappear cleanly there.
    """
    mesh = create_mesh(devices=jax.devices()[:1])
    self.assertEqual(mesh.devices.size, 1)
    self.assertEqual(mesh.shape["data"], 1)

  def test_ici_and_dcn_may_each_be_auto(self):
    """One `-1` per shape is legal, because they resolve against different totals."""
    mesh = create_mesh(ici={"fsdp": -1}, dcn={"data": -1}, devices=jax.devices()[:1])
    self.assertEqual(mesh.devices.size, 1)

  @parameterized.named_parameters(
      # An explicit `-1` with nothing left for it to absorb: fails auto-sizing.
      ("with_auto_data", {"dcn": {"data": -1, "tensor": 2}}, "Cannot auto-size the -1 DCN mesh axis"),
      # Fully specified, so it reaches the product check instead.
      ("fully_specified", {"dcn": {"data": 1, "tensor": 2}}, r"DCN mesh axes .* slice count \(1\)"),
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
      create_mesh(dcn={"data": -1, "fsdp": -1}, devices=jax.devices()[:1])

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

  def test_custom_axis_names(self):
    """A model with its own mesh axes (Wan's `context`) names them; sizes resolve against those axes."""
    axes = ("data", "fsdp", "context", "tensor")
    mesh = create_mesh(ici={"context": -1}, dcn={"context": -1}, devices=jax.devices()[:1], axis_names=axes)
    self.assertEqual(mesh.axis_names, axes)
    self.assertEqual(mesh.devices.shape, (1, 1, 1, 1))
    # The defaults still apply when their axes are on the mesh.
    self.assertEqual(create_mesh(devices=jax.devices()[:1], axis_names=axes).axis_names, axes)

  def test_custom_axis_names_reject_names_outside_them(self):
    with self.assertRaisesRegex(ValueError, r"Unknown ICI mesh axes \['expert'\]"):
      create_mesh(ici={"expert": -1}, devices=jax.devices()[:1], axis_names=("data", "fsdp", "context", "tensor"))

  @parameterized.named_parameters(
      ("ici", {"dcn": {"redundant": -1}}, "default ICI layout puts every device on 'fsdp'"),
      ("dcn", {"ici": {"redundant": -1}}, "default DCN layout puts every device on 'data'"),
  )
  def test_defaults_require_their_axis(self, kwargs, message):
    """`ici` / `dcn` must be given when the default's axis is not on the mesh, rather than guessed."""
    with self.assertRaisesRegex(ValueError, message):
      create_mesh(devices=jax.devices()[:1], axis_names=("redundant", "vae_spatial"), **kwargs)


@pytest.mark.tpu_only
class CreateMeshMultiDeviceTest(parameterized.TestCase):
  """Cases that need a real multi-device mesh.

  Marked `tpu_only` deliberately: unmarked tests are auto-marked `cpu_only` by
  the repo conftest and skipped on accelerator testbeds, which is exactly where
  these need to run.
  """

  def setUp(self):
    super().setUp()
    self.n = _require_multi_device(self)
    self.devices = jax.devices()

  def test_auto_fsdp_absorbs_all_devices(self):
    mesh = create_mesh(devices=self.devices)
    self.assertEqual(mesh.shape["fsdp"], self.n)
    self.assertEqual(mesh.shape["tensor"], 1)

  def test_tensor_parallelism_splits_with_fsdp(self):
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)
    self.assertEqual(mesh.shape["fsdp"], self.n // 2)
    self.assertEqual(mesh.shape["tensor"], 2)

  def test_constrain_shards_activations_across_mesh(self):
    """End to end: logical axes actually split data across devices.

    Asserted via per-device shard shape rather than the `PartitionSpec`. With
    `tensor=2` the mesh is `data=1, fsdp=n/2, tensor=2`, and XLA drops the
    size-1 `data` from the realized spec -- so `("batch", ...)` comes back as
    `P("fsdp", ...)` rather than `P(("data", "fsdp"), ...)`. Shard shape states
    the property we actually care about and does not move when the mesh does.
    """
    sharding = _ExampleSharding()
    fsdp = self.n // 2
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "mlp_mid", ("batch", "length", "mlp")))(jnp.zeros((fsdp, 2, 16)))

    # batch over fsdp, length replicated, mlp 16 over tensor=2.
    self.assertEqual(out.sharding.shard_shape(out.shape), (1, 2, 8))

  def test_replicated_axes_are_not_sharded(self):
    """A rule returning None must leave the dimension whole on every device."""
    sharding = _ExampleSharding()
    fsdp = self.n // 2
    mesh = create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices)

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "hidden", ("batch", "length", "embed")))(jnp.zeros((fsdp, 2, 16)))

    # `embed` is unconstrained for activations, so the last dim stays at 16.
    self.assertEqual(out.sharding.shard_shape(out.shape), (1, 2, 16))

  def test_constrain_shards_activations_under_an_explicit_mesh(self):
    """The explicit path must shard the same way the auto path does.

    This is the case that actually caught the bug. On a single device every
    axis has size 1, so `with_sharding_constraint`'s assertion passes and the
    wrong API goes unnoticed; with real shards it raises.

    The realized spec also differs between modes -- explicit meshes keep the
    size-1 `data` that auto normalizes away -- so this asserts shard shape, which
    is the property that has to match.
    """
    sharding = _ExampleSharding()
    fsdp = self.n // 2
    mesh = _as_explicit(create_mesh(ici={"fsdp": -1, "tensor": 2}, devices=self.devices))

    with jax.sharding.set_mesh(mesh):
      out = jax.jit(lambda x: sharding.constrain(x, "mlp_mid", ("batch", "length", "mlp")))(jnp.zeros((fsdp, 2, 16)))

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
    self.n = _require_multi_device(self)
    self.devices = jax.devices()

  def test_ici_resolves_per_slice_and_dcn_across_slices(self):
    recorded = {}

    def fake_hybrid(ici_shape, dcn_shape, devices, **_):
      recorded["ici"] = tuple(ici_shape)
      recorded["dcn"] = tuple(dcn_shape)
      return np.array(devices).reshape(np.multiply(ici_shape, dcn_shape))

    with mock.patch.object(sharding_lib, "_count_slices", return_value=2):
      with mock.patch.object(sharding_lib.mesh_utils, "create_hybrid_device_mesh", side_effect=fake_hybrid):
        mesh = create_mesh(devices=self.devices)

    # n devices across 2 slices is n/2 per slice. The default `fsdp=-1` must
    # absorb n/2, not n -- absorbing n is the bug this test exists to catch.
    per_slice = self.n // 2
    self.assertEqual(recorded["ici"], (1, per_slice, 1, 1))
    self.assertEqual(recorded["dcn"], (2, 1, 1, 1))
    self.assertEqual(mesh.shape["data"], 2)
    self.assertEqual(mesh.shape["fsdp"], per_slice)

  def test_single_slice_does_not_take_the_hybrid_path(self):
    """`create_hybrid_device_mesh` raises on devices without a `slice_index`.

    It does so even when the DCN shape is all ones, so the branch cannot be
    collapsed into an unconditional hybrid call. This pins that.
    """
    with mock.patch.object(sharding_lib.mesh_utils, "create_hybrid_device_mesh") as hybrid:
      create_mesh(devices=self.devices)
    hybrid.assert_not_called()

  def test_dcn_shape_must_match_the_slice_count(self):
    with mock.patch.object(sharding_lib, "_count_slices", return_value=2):
      with self.assertRaisesRegex(ValueError, r"DCN mesh axes .* slice count \(2\)"):
        create_mesh(dcn={"data": 2 * self.n}, devices=self.devices)

  def test_rejects_devices_that_do_not_divide_into_slices(self):
    """Integer division would silently drop devices; that must raise instead."""
    non_divisor = next(k for k in range(2, self.n + 2) if self.n % k)
    with mock.patch.object(sharding_lib, "_count_slices", return_value=non_divisor):
      with self.assertRaisesRegex(ValueError, "do not divide evenly"):
        create_mesh(devices=self.devices)


if __name__ == "__main__":
  absltest.main()

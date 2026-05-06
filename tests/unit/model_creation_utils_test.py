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

"""Unit tests for model_creation_utils.py."""

import dataclasses
import sys
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax import nnx
from jax.sharding import Mesh
from orbax import checkpoint as ocp
import pytest

from flax.training import train_state
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN, MODEL_MODE_PREFILL
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.model_creation_utils import (
    _align_checkpoint_to_model_shapes,
    _fix_restore_args_for_shape_mismatch,
    _fuse_moe_weights,
    _stored_shape_evenly_shardable,
    _zero_pad_axis,
)
from tests.utils.test_helpers import get_test_config_path


# ---------------------------------------------------------------------------
# Minimal stub for ArrayMetadata (avoids a real Orbax checkpoint on disk).
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _FakeArrayMetadata:
  shape: tuple
  dtype: object = jnp.float32
  sharding: object = None


def _is_fake_meta(x):
  return isinstance(x, _FakeArrayMetadata)


# Monkey-patch the module-level helper so our fake metadata is recognised.
_orig_is_orbax = model_creation_utils._is_orbax_array_metadata  # pylint: disable=protected-access
model_creation_utils._is_orbax_array_metadata = _is_fake_meta  # pylint: disable=protected-access


def _make_restore_arg(global_shape):
  """Return an ArrayRestoreArgs with a trivial NamedSharding."""
  mesh = jax.sharding.Mesh(jax.local_devices()[:1], ("x",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return ocp.ArrayRestoreArgs(
      global_shape=global_shape,
      shape=global_shape,
      sharding=sharding,
      dtype=jnp.float32,
  )


def _make_config(**kwargs):
  """Returns a minimal pyconfig suitable for model-creation tests."""
  defaults = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "base_num_decoder_layers": 2,
      "attention": "dot_product",
      "max_target_length": 16,
      "base_emb_dim": 32,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "max_prefill_predict_length": 4,
  }
  defaults.update(kwargs)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      **defaults,
  )


def _make_mesh(config):
  devices_array = maxtext_utils.create_device_mesh(config)
  return Mesh(devices_array, config.mesh_axes)


@pytest.mark.tpu_only
class FixRestoreArgsRankGuardTest(unittest.TestCase):
  """_fix_restore_args_for_shape_mismatch must dispatch correctly on rank/divisibility.

  Two distinct outcomes for a same-rank shape mismatch:
    - stored shape divides evenly across the original NamedSharding's mesh axes:
      keep the model's sharding and load at stored shape (Orbax sees each device's
      local slice — no replicated fanout). global_shape becomes stored_shape.
    - otherwise: fall back to fully-replicated load (global_shape cleared).
  Rank mismatch is short-circuited unchanged regardless.
  """

  def setUp(self):
    self.mesh = jax.sharding.Mesh(jax.local_devices()[:1], ("x",))

  def _run_fix(self, stored_shape, model_shape, sharding=None):
    arg = _make_restore_arg(model_shape)
    if sharding is not None:
      arg = dataclasses.replace(arg, sharding=sharding)
    restore_args = {"kernel": arg}
    metadata_tree = {"kernel": _FakeArrayMetadata(shape=stored_shape)}
    return _fix_restore_args_for_shape_mismatch(restore_args, metadata_tree, self.mesh)

  def test_scanned_ckpt_unscanned_model_raises_error(self):
    """Rank mismatch (scanned ckpt rank 4 vs unscanned model rank 3): arg must be unchanged."""
    # Simulates: scanned checkpoint key kernel (94, 256, 4, 128) vs vLLM model (256, 64, 128).
    stored_shape = (94, 256, 4, 128)
    model_shape = (256, 64, 128)
    with self.assertRaises(ValueError) as cm:
      self._run_fix(stored_shape, model_shape)
    self.assertIn("Checkpoint rank mismatches detected", str(cm.exception))

  def test_same_rank_shape_mismatch_divisible_keeps_sharding(self):
    """Same rank, shape mismatch, stored shape divisible by sharding: keep sharding at stored shape."""
    # 1-device mesh trivially divides any stored shape, so this hits the sharded path.
    stored_shape = (256, 4, 128)
    model_shape = (256, 64, 128)
    fixed = self._run_fix(stored_shape, model_shape)
    arg = fixed["kernel"]
    # New path: global_shape carries the stored shape; original sharding is preserved.
    self.assertEqual(arg.global_shape, stored_shape)
    self.assertIsInstance(arg.sharding, jax.sharding.NamedSharding)

  def test_same_rank_shape_mismatch_indivisible_falls_back_to_replicated(self):
    """Stored dim not divisible by mesh axis size: fall back to fully-replicated."""
    if jax.device_count() < 2:
      self.skipTest("Requires >=2 devices to construct an indivisible mesh axis.")
    devices = jax.local_devices()[:2]
    multi_mesh = jax.sharding.Mesh(devices, ("x",))
    sharding = jax.sharding.NamedSharding(multi_mesh, jax.sharding.PartitionSpec(None, "x", None))
    # stored axis 1 = 3, mesh "x" = 2 -> 3 % 2 != 0 -> can't shard, must replicate.
    stored_shape = (256, 3, 128)
    model_shape = (256, 4, 128)
    arg = dataclasses.replace(_make_restore_arg(model_shape), sharding=sharding)
    fixed = _fix_restore_args_for_shape_mismatch(
        {"kernel": arg}, {"kernel": _FakeArrayMetadata(shape=stored_shape)}, multi_mesh
    )
    arg = fixed["kernel"]
    # Replicated fallback: global_shape cleared, sharding swapped to fully-replicated.
    self.assertIsNone(arg.global_shape)
    self.assertEqual(arg.sharding.spec, jax.sharding.PartitionSpec())

  def test_same_shape_no_modification(self):
    """Identical shapes: arg must be unchanged."""
    shape = (4096, 4, 128)
    fixed = self._run_fix(shape, shape)
    arg = fixed["kernel"]
    self.assertEqual(arg.global_shape, shape)

  def test_scanned_both_same_rank_shape_mismatch_divisible_keeps_sharding(self):
    """Scanned ckpt + scanned model + KV padding, divisible stored shape: keep sharding."""
    stored_shape = (94, 256, 4, 128)
    model_shape = (94, 256, 64, 128)
    fixed = self._run_fix(stored_shape, model_shape)
    arg = fixed["kernel"]
    self.assertEqual(arg.global_shape, stored_shape)


@pytest.mark.tpu_only
class TestAlignCheckpointToModelShapes(unittest.TestCase):
  """_align_checkpoint_to_model_shapes must dispatch on logical axis names.

  Two distinct semantics must be preserved:
    - kv_heads → jnp.repeat (each KV head is replicated across query heads on TP shards).
    - mlp_moe / activation_mlp → jnp.pad with zeros at the end (MoE GMM_v2 kernel padding).
  """

  def test_no_op_when_shapes_match(self):
    """Identical shapes: arr is returned unchanged (modulo device_put)."""
    ckpt = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    model = jnp.zeros((2, 4), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("a", "b"))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ckpt))

  def test_kv_heads_repeat_layout(self):
    """KV-head axis must use jnp.repeat: [h0, h0, h1, h1], NOT [h0, h1, h0, h1]."""
    # Distinguishable per-head values so the layout assertion is unambiguous.
    ckpt = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],  # embed=0: kv_head_0=[1,1], kv_head_1=[2,2]
            [[3.0, 3.0], [4.0, 4.0]],  # embed=1: kv_head_0=[3,3], kv_head_1=[4,4]
        ],
        dtype=jnp.float32,
    )
    model = jnp.zeros((2, 4, 2), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("embed", "kv_heads", "kv_head_dim"))
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 4, 2))
    # kv_heads axis layout for embed=0 must be [h0, h0, h1, h1]:
    np.testing.assert_array_equal(out_np[0, 0], [1.0, 1.0])
    np.testing.assert_array_equal(out_np[0, 1], [1.0, 1.0])
    np.testing.assert_array_equal(out_np[0, 2], [2.0, 2.0])
    np.testing.assert_array_equal(out_np[0, 3], [2.0, 2.0])

  def test_mlp_moe_zero_pads_at_end_wi_0(self):
    """mlp_moe axis must zero-pad at the end (not repeat) — wi_0/wi_1 last-axis case."""
    ckpt = jnp.ones((2, 4, 6), dtype=jnp.float32)  # exp=2, embed=4, mlp=6
    model = jnp.zeros((2, 4, 8), dtype=jnp.float32)  # mlp padded to 8
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("exp", "embed_moe", "mlp_moe"))
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 4, 8))
    # First 6 cols preserved as ones; last 2 cols are zeros.
    np.testing.assert_array_equal(out_np[..., :6], np.ones((2, 4, 6), dtype=np.float32))
    np.testing.assert_array_equal(out_np[..., 6:], np.zeros((2, 4, 2), dtype=np.float32))

  def test_mlp_moe_zero_pads_at_end_wo(self):
    """wo has the MLP dim on axis 1 — zero-pad at the end of axis 1."""
    ckpt = jnp.ones((2, 6, 4), dtype=jnp.float32)
    model = jnp.zeros((2, 8, 4), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("exp", "mlp_moe", "embed_moe"))
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 8, 4))
    np.testing.assert_array_equal(out_np[:, :6, :], np.ones((2, 6, 4), dtype=np.float32))
    np.testing.assert_array_equal(out_np[:, 6:, :], np.zeros((2, 2, 4), dtype=np.float32))

  def test_activation_mlp_zero_pads_bias(self):
    """activation_mlp axis (bias arrays) must also zero-pad at the end."""
    ckpt = jnp.ones((2, 6), dtype=jnp.float32)
    model = jnp.zeros((2, 8), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("exp", "activation_mlp"))
    out_np = np.asarray(out)
    np.testing.assert_array_equal(out_np[:, :6], np.ones((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(out_np[:, 6:], np.zeros((2, 2), dtype=np.float32))

  def test_unknown_axis_divisible_falls_back_to_repeat(self):
    """Unknown axis with divisible dims preserves prior repeat behavior (backward compat)."""
    ckpt = jnp.array([1.0, 2.0], dtype=jnp.float32)
    model = jnp.zeros((4,), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, ("unknown_axis",))
    np.testing.assert_array_equal(np.asarray(out), [1.0, 1.0, 2.0, 2.0])

  def test_unknown_axis_non_divisible_raises(self):
    """Unknown axis with non-divisible dims must raise a clear error pointing at the axis."""
    ckpt = jnp.ones((3,), dtype=jnp.float32)
    model = jnp.zeros((4,), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "not registered in _VLLM_REPEAT_AXES"):
      _align_checkpoint_to_model_shapes(ckpt, model, ("unknown_axis",))

  def test_logical_axes_none_falls_back_to_repeat(self):
    """When logical_axes is None, divisible dims default to repeat (legacy behavior)."""
    ckpt = jnp.array([1.0, 2.0], dtype=jnp.float32)
    model = jnp.zeros((4,), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, None)
    np.testing.assert_array_equal(np.asarray(out), [1.0, 1.0, 2.0, 2.0])

  def test_kv_heads_non_divisible_raises(self):
    """kv_heads axis with non-divisible dims raises (jnp.repeat is undefined here)."""
    ckpt = jnp.ones((4, 3, 2), dtype=jnp.float32)
    model = jnp.zeros((4, 4, 2), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "kv_heads"):
      _align_checkpoint_to_model_shapes(ckpt, model, ("embed", "kv_heads", "kv_head_dim"))

  def test_rank_mismatch_raises(self):
    """Different ranks (scanned vs unscanned ckpt) must raise the helpful message."""
    ckpt = jnp.ones((4, 4, 2), dtype=jnp.float32)
    model = jnp.zeros((2, 4, 4, 2), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "different ranks"):
      _align_checkpoint_to_model_shapes(ckpt, model, ("scan", "embed", "kv_heads", "kv_head_dim"))


@pytest.mark.tpu_only
class TestFuseMoeWeights(unittest.TestCase):
  """_fuse_moe_weights must zero-pad each half BEFORE concatenation when MLP-dim padding applies.

  Otherwise zeros land between wi_0's data and wi_1's data, corrupting the boundary.
  """

  def test_no_op_when_model_has_unfused_wi(self):
    """If model has wi_0/wi_1 (not fused), the helper returns the ckpt tree unchanged."""
    ckpt = {"wi_0": np.ones((2, 4, 3), dtype=np.float32), "wi_1": 2 * np.ones((2, 4, 3), dtype=np.float32)}
    model = {"wi_0": np.zeros((2, 4, 3), dtype=np.float32), "wi_1": np.zeros((2, 4, 3), dtype=np.float32)}
    out = _fuse_moe_weights(ckpt, model)
    self.assertIn("wi_0", out)
    self.assertIn("wi_1", out)
    self.assertNotIn("wi", out)

  def test_fuse_without_mlp_padding(self):
    """When ckpt and model halves match, simple concat (no padding inserted)."""
    wi_0 = np.ones((2, 4, 3), dtype=np.float32)
    wi_1 = 2 * np.ones((2, 4, 3), dtype=np.float32)
    ckpt = {"wi_0": wi_0, "wi_1": wi_1}
    # Model has fused wi of last-dim 6 = 2 * 3 (no MLP padding).
    model = {"wi": np.zeros((2, 4, 6), dtype=np.float32)}
    out = _fuse_moe_weights(ckpt, model)
    self.assertEqual(out["wi"].shape, (2, 4, 6))
    np.testing.assert_array_equal(out["wi"][..., :3], wi_0)
    np.testing.assert_array_equal(out["wi"][..., 3:], wi_1)

  def test_fuse_with_mlp_padding_pads_each_half(self):
    """With MLP padding, each half must be zero-padded BEFORE concatenation.

    Result must be ``[wi_0_data | 0 | wi_1_data | 0]`` (zeros at the end of each half),
    NOT ``[wi_0_data | wi_1_data | 0 | 0]`` (zeros only at the very tail).
    """
    wi_0 = np.ones((2, 4, 3), dtype=np.float32)
    wi_1 = 2 * np.ones((2, 4, 3), dtype=np.float32)
    ckpt = {"wi_0": wi_0, "wi_1": wi_1}
    # Model has fused wi of last-dim 8 = 2 * 4 (each half padded from 3 → 4).
    model = {"wi": np.zeros((2, 4, 8), dtype=np.float32)}
    out = _fuse_moe_weights(ckpt, model)
    self.assertEqual(out["wi"].shape, (2, 4, 8))
    # First half: wi_0_data (3 cols) then 1 col of zeros.
    np.testing.assert_array_equal(out["wi"][..., :3], wi_0)
    np.testing.assert_array_equal(out["wi"][..., 3:4], np.zeros((2, 4, 1), dtype=np.float32))
    # Second half: wi_1_data (3 cols) then 1 col of zeros.
    np.testing.assert_array_equal(out["wi"][..., 4:7], wi_1)
    np.testing.assert_array_equal(out["wi"][..., 7:8], np.zeros((2, 4, 1), dtype=np.float32))

  def test_recurses_through_nested_dicts(self):
    """The helper should walk arbitrary nested dicts and only fuse at MoE blocks."""
    wi_0 = np.ones((2, 4, 3), dtype=np.float32)
    wi_1 = 2 * np.ones((2, 4, 3), dtype=np.float32)
    other = np.full((2, 4), 7.0, dtype=np.float32)
    ckpt = {"layers": {"0": {"moe": {"wi_0": wi_0, "wi_1": wi_1}, "other": other}}}
    model = {"layers": {"0": {"moe": {"wi": np.zeros((2, 4, 6), dtype=np.float32)}, "other": other}}}
    out = _fuse_moe_weights(ckpt, model)
    self.assertEqual(out["layers"]["0"]["moe"]["wi"].shape, (2, 4, 6))
    np.testing.assert_array_equal(out["layers"]["0"]["other"], other)


@pytest.mark.tpu_only
class TestShardedZeroPad(unittest.TestCase):
  """Covers the per-shard zero-pad path that avoids the replicated-fanout OOM.

  Without this path, large MoE checkpoints loaded through the shape-mismatch
  fallback in ``_fix_restore_args_for_shape_mismatch`` were forced to land as
  full replicas on every TP device. ``_stored_shape_evenly_shardable`` decides
  when the model's NamedSharding can carry the stored shape directly (no
  replication), and ``_zero_pad_axis`` then expands each local shard via
  ``shard_map`` instead of materializing a global replicated tensor.
  """

  def test_evenly_shardable_true_when_stored_dim_divides_mesh_axis(self):
    devices = jax.local_devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("model",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model", None))
    restore_arg = _make_restore_arg((128, 1024, 2048))
    restore_arg = dataclasses.replace(restore_arg, sharding=sharding)
    # 1-device mesh: any stored dim trivially divides 1.
    self.assertTrue(_stored_shape_evenly_shardable(restore_arg, (128, 768, 2048)))

  def test_evenly_shardable_false_when_stored_dim_indivisible(self):
    if jax.device_count() < 2:
      self.skipTest("Requires >=2 devices to construct a non-trivial mesh size.")
    devices = jax.local_devices()[:2]
    mesh = jax.sharding.Mesh(devices, ("model",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model", None))
    restore_arg = _make_restore_arg((128, 8, 2048))
    restore_arg = dataclasses.replace(restore_arg, sharding=sharding)
    # stored_shape[1]=3 is not divisible by mesh "model"=2 -> must fall back to replicated.
    self.assertFalse(_stored_shape_evenly_shardable(restore_arg, (128, 3, 2048)))

  def test_zero_pad_axis_no_op_when_extra_zero(self):
    arr = jnp.ones((2, 4, 2), dtype=jnp.float32)
    self.assertIs(_zero_pad_axis(arr, axis=1, extra=0), arr)

  def test_zero_pad_axis_unsharded_falls_back_to_global_pad(self):
    # Unsharded input -> jnp.pad (zeros at the global tail of the axis).
    arr = jnp.ones((2, 4, 2), dtype=jnp.float32)
    out = _zero_pad_axis(arr, axis=1, extra=2)
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 6, 2))
    np.testing.assert_array_equal(out_np[:, :4, :], np.ones((2, 4, 2), dtype=np.float32))
    np.testing.assert_array_equal(out_np[:, 4:, :], np.zeros((2, 2, 2), dtype=np.float32))

  def test_zero_pad_axis_sharded_pads_each_local_shard(self):
    """Per-shard layout: zeros land at the tail of *each* shard, not the global tail.

    Mathematically equivalent to the global-tail layout under matmul along the
    padded axis with a matching pad on the consuming weight (the MoE wi/wo + GMM_v2
    case). The point of this test is to lock in the *layout invariant*: per-shard
    tails are zero, per-shard heads preserve the original ckpt slice.
    """
    if jax.device_count() < 4:
      self.skipTest("Requires >=4 devices to exercise the sharded pad path with TP=4.")
    devices = jax.local_devices()[:4]
    mesh = jax.sharding.Mesh(devices, ("model",))
    spec = jax.sharding.PartitionSpec(None, "model", None)
    sharding = jax.sharding.NamedSharding(mesh, spec)

    ckpt_np = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
    arr = jax.device_put(jnp.asarray(ckpt_np), sharding)

    out = _zero_pad_axis(arr, axis=1, extra=4)  # 8 -> 12, 4 shards -> +1 zero per shard
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 12, 2))

    for s in range(4):
      # Per-shard head holds the original ckpt slice [s*2:s*2+2].
      for i in range(2):
        np.testing.assert_array_equal(out_np[:, s * 3 + i, :], ckpt_np[:, s * 2 + i, :])
      # Per-shard tail is zero.
      np.testing.assert_array_equal(out_np[:, s * 3 + 2, :], np.zeros((2, 2), dtype=np.float32))


@pytest.mark.tpu_only
class TestPartitionSpecUnwrapForAlignment(unittest.TestCase):
  """Locks in the runtime extraction path used by from_pretrained.

  ``nnx.get_partition_spec`` returns a tree whose leaves are ``nnx.Variable``
  objects wrapping ``PartitionSpec`` values, not raw ``PartitionSpec``s.
  Without unwrapping, ``_normalize_logical_axes`` returns ``None`` for every
  leaf, the per-axis dispatch falls back to divisibility-based repeat, and MoE
  MLP-dim padding (e.g. ``mlp_moe`` 768 → 1024) raises because 1024 is not a
  multiple of 768. This test fails on the unwrapped value path and asserts
  ``_align_checkpoint_to_model_shapes`` then dispatches into ``_ZERO_PAD_AXES``.
  """

  def test_unwrap_yields_partition_spec_and_enables_zero_pad(self):
    class _TinyMoE(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.wo = nnx.Param(
            jax.random.normal(rngs.params(), (2, 6, 4), dtype=jnp.float32),
            sharding=("exp", "mlp_moe", "embed_moe"),
        )

    # flax >= 0.12 requires an active mesh + logical_axis_rules to construct
    # a Variable annotated with logical sharding names.
    devices = jax.local_devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("x",), axis_types=(jax.sharding.AxisType.Explicit,))
    rules = (("exp", None), ("mlp_moe", None), ("embed_moe", None))
    with jax.sharding.set_mesh(mesh), nn.logical_axis_rules(rules):
      module = nnx.eval_shape(lambda: _TinyMoE(nnx.Rngs(params=0)))

    _, abstract_state = nnx.split(module)
    specs = nnx.get_partition_spec(abstract_state)

    # Mirror the unwrap done in from_pretrained.
    logical_axes_tree = jax.tree.map(
        lambda v: v.get_value(),
        specs,
        is_leaf=lambda n: isinstance(n, nnx.Variable),
    )

    # Walk to the wo leaf (nnx.State is dict-like).
    wo_axes = logical_axes_tree["wo"]
    self.assertIsInstance(wo_axes, jax.sharding.PartitionSpec)
    self.assertNotIsInstance(wo_axes, nnx.Variable)
    self.assertEqual(tuple(wo_axes), ("exp", "mlp_moe", "embed_moe"))

    # End-to-end: leaf must drive zero-pad dispatch on axis 1 (mlp_moe).
    ckpt = jnp.ones((2, 4, 4), dtype=jnp.float32)
    model = jnp.zeros((2, 6, 4), dtype=jnp.float32)
    out = _align_checkpoint_to_model_shapes(ckpt, model, wo_axes)
    out_np = np.asarray(out)
    self.assertEqual(out_np.shape, (2, 6, 4))
    np.testing.assert_array_equal(out_np[:, :4, :], np.ones((2, 4, 4), dtype=np.float32))
    np.testing.assert_array_equal(out_np[:, 4:, :], np.zeros((2, 2, 4), dtype=np.float32))


class TestGetTransformerModel(unittest.TestCase):
  """Tests for get_transformer_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_linen_module_when_rngs_is_none(self):
    """Without rngs, should return a Linen nn.Module."""
    model = model_creation_utils.get_transformer_model(self.config, self.mesh, quant=None, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_returns_nnx_module_when_rngs_provided(self):
    """With rngs, should return an NNX nnx.Module."""
    model = nnx.eval_shape(
        lambda: model_creation_utils.get_transformer_model(
            self.config, self.mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=1, aqt=2)
        )
    )
    self.assertIsInstance(model, nnx.Module)

  def test_respects_model_mode_prefill(self):
    """Linen model created with MODEL_MODE_PREFILL should differ from train mode."""
    linen_train = model_creation_utils.get_transformer_model(
        self.config, self.mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=None
    )
    linen_prefill = model_creation_utils.get_transformer_model(
        self.config, self.mesh, quant=None, model_mode=MODEL_MODE_PREFILL, rngs=None
    )
    # Both are still nn.Module instances
    self.assertIsInstance(linen_train, nn.Module)
    self.assertIsInstance(linen_prefill, nn.Module)


class TestCreateModel(unittest.TestCase):
  """Tests for create_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_linen_model_without_rngs(self):
    model = model_creation_utils.create_model(self.config, self.mesh)
    self.assertIsInstance(model, nn.Module)

  def test_returns_nnx_model_with_rngs(self):
    model = nnx.eval_shape(
        lambda: model_creation_utils.create_model(self.config, self.mesh, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    )
    self.assertIsInstance(model, nnx.Module)

  def test_model_mode_train_default(self):
    """Default model_mode is MODEL_MODE_TRAIN."""
    model = model_creation_utils.create_model(self.config, self.mesh)
    self.assertIsInstance(model, nn.Module)


class TestFromConfig(unittest.TestCase):
  """Tests for from_config()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_linen_path_rngs_none(self):
    """from_config with rngs=None should return a Linen nn.Module."""
    model = model_creation_utils.from_config(self.config, mesh=self.mesh, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_nnx_path_with_rngs(self):
    """from_config with rngs provided should return an NNX nnx.Module."""
    model = nnx.eval_shape(
        lambda: model_creation_utils.from_config(self.config, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    )
    self.assertIsInstance(model, nnx.Module)

  def test_mesh_created_from_devices_when_none(self):
    """from_config should work when mesh is None (creates mesh internally)."""
    model = model_creation_utils.from_config(self.config, devices=None, mesh=None, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_model_mode_is_forwarded(self):
    """from_config should accept and forward model_mode."""
    model = model_creation_utils.from_config(self.config, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_explicit_shard_mode_creates_mesh_with_explicit_axis_types(self):
    """from_config with shard_mode=explicit should create mesh using AxisType.Explicit."""
    cfg = _make_config(shard_mode="explicit")
    # Should not raise; mesh is built with AxisType.Explicit for each axis
    model = model_creation_utils.from_config(cfg, mesh=None, rngs=None)
    self.assertIsInstance(model, nn.Module)


class TestCreateNNXAbstractModel(unittest.TestCase):
  """Tests for create_nnx_abstract_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_tuple_of_callable_and_module(self):
    create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    self.assertTrue(callable(create_fn))
    self.assertIsInstance(abstract_model, nnx.Module)

  def test_abstract_model_has_abstract_arrays(self):
    """Abstract model leaves should be ShapeDtypeStruct, not concrete arrays."""
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    _, state = nnx.split(abstract_model)
    leaves = jax.tree.leaves(state)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      # In abstract state, values are nnx.Variable wrapping abstract shapes/ShapeDtypeStruct
      # Concrete jax.Array would have a .devices() method; abstract ones should not be Arrays
      self.assertNotIsInstance(leaf, jax.Array)

  def test_create_fn_produces_concrete_model(self):
    """The returned create_fn should produce a real (concrete) NNX Module."""
    create_fn, _ = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    with self.mesh:
      concrete = create_fn()
    self.assertIsInstance(concrete, nnx.Module)
    leaves = jax.tree.leaves(nnx.state(concrete))
    for leaf in leaves:
      self.assertIsInstance(leaf, jax.Array)

  def test_works_without_explicit_mesh(self):
    """create_nnx_abstract_model should work when mesh=None (from_config creates mesh)."""
    create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=None)
    self.assertTrue(callable(create_fn))
    self.assertIsInstance(abstract_model, nnx.Module)

  def test_explicit_rng_key_is_used(self):
    """Passing a rng_key should not raise and returns valid abstract model."""
    rng_key = jax.random.PRNGKey(42)
    create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(
        self.config, mesh=self.mesh, rng_key=rng_key
    )
    self.assertTrue(callable(create_fn))
    self.assertIsInstance(abstract_model, nnx.Module)

  def test_prefill_model_mode(self):
    """create_nnx_abstract_model should accept MODEL_MODE_PREFILL."""
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(
        self.config, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL
    )
    self.assertIsInstance(abstract_model, nnx.Module)


class TestCreateNnxModel(unittest.TestCase):
  """Tests for from_pretrained()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_no_checkpoint_returns_model_and_mesh(self):
    """Without load_parameters_path, should return the model cleanly."""
    model = model_creation_utils.from_pretrained(self.config, self.mesh)
    self.assertIsInstance(model, models.Transformer)

  def test_mesh_none_uses_abstract_model_mesh(self):
    """When mesh=None is passed, the function resolves it from the abstract model."""
    model, mesh = model_creation_utils.from_pretrained(self.config, mesh=None)
    self.assertIsInstance(model, models.Transformer)
    self.assertIsInstance(mesh, Mesh)

  def test_explicit_rng_key(self):
    """An explicit rng_key should be accepted without error."""
    rng_key = jax.random.PRNGKey(99)
    model = model_creation_utils.from_pretrained(self.config, self.mesh, rng_key=rng_key)
    self.assertIsInstance(model, models.Transformer)

  def test_inference_mode_disables_dropout_rng(self):
    """MODEL_MODE_PREFILL should create rngs without a dropout key."""
    model = model_creation_utils.from_pretrained(self.config, self.mesh, model_mode=MODEL_MODE_PREFILL)
    self.assertIsInstance(model, models.Transformer)

  def test_debug_sharding_flag(self):
    """debug_sharding=True should execute the sharding-print path without error."""
    cfg = _make_config(debug_sharding=True)
    model = model_creation_utils.from_pretrained(cfg, self.mesh)
    self.assertIsInstance(model, models.Transformer)

  # ---- checkpoint loading: mocked paths ----

  def _make_linen_metadata_mock(self):
    """Mock ocp metadata that looks like a Linen checkpoint."""
    meta = MagicMock()
    meta.item_metadata.tree.keys.return_value = ["params"]
    meta.item_metadata.tree.get.return_value = {"params": {}}
    return meta

  def _make_nnx_metadata_mock(self):
    """Mock ocp metadata that looks like an NNX checkpoint."""
    meta = MagicMock()
    meta.item_metadata.tree.keys.return_value = ["decoder"]
    meta.item_metadata.tree.get.return_value = {}
    return meta

  @patch("maxtext.utils.model_creation_utils.ocp")
  def test_load_nnx_checkpoint(self, mock_ocp):
    """NNX-format checkpoint: restored values are wrapped under a 'value' key."""
    # Echo back the `item` argument passed by from_pretrained to ckptr.restore.
    # For NNX checkpoints, item IS already {leaf: {"value": array}, ...}, so
    # returning it directly gives a correctly-structured restored dict that
    # matches the model's own state — regardless of the exact leaf count.
    mock_ckptr = MagicMock()
    mock_ckptr.metadata.return_value = self._make_nnx_metadata_mock()
    mock_ckptr.restore.side_effect = lambda path, item=None, **kw: item
    mock_ocp.Checkpointer.return_value = mock_ckptr
    mock_ocp.PyTreeCheckpointHandler.return_value = MagicMock()
    mock_ocp.checkpoint_utils.construct_restore_args.return_value = {}
    mock_ocp.ArrayRestoreArgs = ocp.ArrayRestoreArgs

    cfg = _make_config(enable_checkpointing=True, load_parameters_path="gs://fake/nnx_ckpt")
    model = model_creation_utils.from_pretrained(cfg, self.mesh)
    self.assertIsInstance(model, models.Transformer)

  @patch("maxtext.utils.model_creation_utils.ocp")
  def test_load_linen_checkpoint(self, mock_ocp):
    """Linen-format checkpoint: restored values are nested under 'params'/'params'."""
    # Echo back the `item` argument passed by from_pretrained to ckptr.restore.
    # For Linen checkpoints, item IS already {"params": {"params": arrays}}, so
    # returning it directly gives a correctly-structured restored dict that
    # matches the model's own state — regardless of the exact leaf count.
    mock_ckptr = MagicMock()
    mock_ckptr.metadata.return_value = self._make_linen_metadata_mock()
    mock_ckptr.restore.side_effect = lambda path, item=None, **kw: item
    mock_ocp.Checkpointer.return_value = mock_ckptr
    mock_ocp.PyTreeCheckpointHandler.return_value = MagicMock()
    mock_ocp.checkpoint_utils.construct_restore_args.return_value = {}
    mock_ocp.ArrayRestoreArgs = ocp.ArrayRestoreArgs

    cfg = _make_config(enable_checkpointing=True, load_parameters_path="gs://fake/linen_ckpt")
    model = model_creation_utils.from_pretrained(cfg, self.mesh)
    self.assertIsInstance(model, models.Transformer)

  @patch("maxtext.utils.model_creation_utils.ocp")
  def test_checkpoint_load_error_raises_value_error(self, mock_ocp):
    """Any exception during checkpoint loading should be re-raised as ValueError."""
    mock_ckptr = MagicMock()
    mock_ckptr.metadata.side_effect = RuntimeError("disk on fire")
    mock_ocp.Checkpointer.return_value = mock_ckptr
    mock_ocp.PyTreeCheckpointHandler.return_value = MagicMock()

    cfg = _make_config(enable_checkpointing=True, load_parameters_path="gs://fake/bad_ckpt")
    with self.assertRaises(ValueError):
      model_creation_utils.from_pretrained(cfg, self.mesh)


class TestSetupDecodeStateFromNnx(unittest.TestCase):
  """Tests for setup_decode_state_from_nnx()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)
    self.rng = jax.random.PRNGKey(0)

  def test_returns_linen_train_state_and_annotations(self):
    """Should return a linen TrainState whose params mirror the NNX model's nnx.Param values."""
    # Build a real (small) NNX model WITHOUT any patch active so from_pretrained
    # runs normally and produces concrete jax.Array weights.
    real_nnx_model = model_creation_utils.from_pretrained(self.config, mesh=self.mesh)

    linen_model = model_creation_utils.from_config(self.config, mesh=self.mesh, rngs=None)

    # Now patch from_pretrained so setup_decode_state_from_nnx never touches a checkpoint.
    with patch("maxtext.utils.model_creation_utils.from_pretrained", return_value=real_nnx_model) as mock_fp:
      state, state_mesh_annotations = model_creation_utils.setup_decode_state_from_nnx(
          linen_model, self.config, self.rng, self.mesh
      )

    # from_pretrained must have been called with the right model_mode.
    mock_fp.assert_called_once()
    _, call_kwargs = mock_fp.call_args
    self.assertEqual(call_kwargs.get("model_mode"), MODEL_MODE_AUTOREGRESSIVE)

    # The result should be a linen TrainState.
    self.assertIsInstance(state, train_state.TrainState)

    # Params must be nested under "params" and be non-empty concrete arrays.
    self.assertIn("params", state.params)
    param_leaves = jax.tree.leaves(state.params["params"])
    self.assertGreater(len(param_leaves), 0)
    for leaf in param_leaves:
      self.assertIsInstance(leaf, jax.Array)

    # The NNX Param values and the extracted linen params must be numerically identical.
    nnx_param_state = nnx.state(real_nnx_model, nnx.Param)
    nnx_leaves = jax.tree.leaves(nnx_param_state)
    linen_leaves = jax.tree.leaves(state.params["params"])
    self.assertEqual(len(nnx_leaves), len(linen_leaves))
    for nnx_val, linen_val in zip(nnx_leaves, linen_leaves):
      self.assertTrue(jnp.all(nnx_val == linen_val))

    # state_mesh_annotations must be returned (non-None).
    self.assertIsNotNone(state_mesh_annotations)


if __name__ == "__main__":
  unittest.main()

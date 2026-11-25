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

"""Unit tests for `maxtext.integration.vllm.scanned_checkpoint_loader` (CPU-only)."""

import os

# Must precede the first JAX/Flax import so CPU backend and multi-device count
# take effect when executed directly via `python` or in the 4-device child process.
_REQUIRED_DEVICES = 4
_CHILD_ENV = "_MAXTEXT_SCANNED_CKPT_LOADER_CHILD"
_IN_CHILD = os.environ.get(_CHILD_ENV) == "1"
if _IN_CHILD or "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = (
      f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}"
  ).strip()
if _IN_CHILD:
  os.environ["JAX_PLATFORMS"] = "cpu"
else:
  os.environ.setdefault("JAX_PLATFORMS", "cpu")

import re
import subprocess
import sys
import types as pytypes
from typing import NamedTuple
import unittest
from unittest import mock
import warnings

from absl import app
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as ocp
import pytest

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


def _import_scanned_checkpoint_loader():
  """Imports `scanned_checkpoint_loader`, stubbing `model_creation_utils` only if optional TF/Orbax deps fail."""
  mod_name = "maxtext.utils.model_creation_utils"
  try:
    from maxtext.integration.vllm import scanned_checkpoint_loader as loader  # pylint: disable=import-outside-toplevel

    return loader
  except Exception:  # pylint: disable=broad-exception-caught
    if mod_name in sys.modules:
      raise
    import maxtext.utils as maxtext_utils_pkg  # pylint: disable=import-outside-toplevel

    stub = pytypes.ModuleType(mod_name)
    stub.from_pretrained = None
    with (
        mock.patch.dict(sys.modules, {mod_name: stub}),
        mock.patch.object(maxtext_utils_pkg, "model_creation_utils", stub, create=True),
    ):
      from maxtext.integration.vllm import scanned_checkpoint_loader as loader  # pylint: disable=import-outside-toplevel

      return loader


scanned_checkpoint_loader = _import_scanned_checkpoint_loader()

pytestmark = pytest.mark.post_training


class _Group(NamedTuple):
  """Minimal plan-entry / plan-group stub for hook orchestration unit tests."""

  op: str
  source_keys: tuple[tuple[str, ...], ...]
  targets: tuple[tuple[str, ...], ...]
  local_index: int | None = None


_WO_GROUP = _Group("slice", (("layers", "wo"),), (("layers_0", "wo"), ("layers_1", "wo")))
_WI_GROUP = _Group(
    "fuse_moe",
    (("layers", "wi_0"), ("layers", "wi_1")),
    (("layers_0", "wi"), ("layers_1", "wi")),
)
_SHARED_LOCAL_GROUPS = (
    _Group(
        "slice",
        (("layers", "local_layers", "wo"),),
        (("layers_0", "wo"), ("layers_2", "wo")),
        local_index=0,
    ),
    _Group(
        "slice",
        (("layers", "local_layers", "wo"),),
        (("layers_1", "wo"), ("layers_3", "wo")),
        local_index=1,
    ),
    _Group(
        "fuse_moe",
        (("layers", "local_layers", "wi_0"), ("layers", "local_layers", "wi_1")),
        (("layers_0", "wi"), ("layers_2", "wi")),
        local_index=0,
    ),
    _Group(
        "fuse_moe",
        (("layers", "local_layers", "wi_0"), ("layers", "local_layers", "wi_1")),
        (("layers_1", "wi"), ("layers_3", "wi")),
        local_index=1,
    ),
)


class _FakeConverter:
  """Stands in for `MaxTextToMaxTextConverter` in hook-level unit tests.

  `convert` plans with `_build_plan` and runs each group through `_execute_group`,
  the two hooks `load_and_convert_scanned_checkpoint` wraps.
  """

  instances: list["_FakeConverter"] = []
  groups: tuple[_Group, ...] = (_WO_GROUP, _WI_GROUP)
  watched: jax.Array | None = None

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.planned_sources = []
    self.watched_alive_during_fuse = None
    self.fuse_inputs = []
    _FakeConverter.instances.append(self)

  def _build_plan(self, src_flat, tgt_flat, skip_paths):
    """Records planned source keys and returns configured test groups."""
    del tgt_flat, skip_paths
    self.planned_sources = sorted(src_flat)
    return self.groups

  def _execute_group(self, group, src_flat, tgt_flat):
    """Executes a stub group while tracking source liveness during `fuse_moe`."""
    del tgt_flat
    raw_sources = [src_flat[key] for key in group.source_keys]
    if group.op == "fuse_moe":
      self.fuse_inputs.append(tuple(raw_sources))
      if self.watched is not None:
        alive = not self.watched.is_deleted()
        self.watched_alive_during_fuse = (
            alive if self.watched_alive_during_fuse is None else self.watched_alive_during_fuse and alive
        )
    sources = [arr[:, group.local_index] for arr in raw_sources] if group.local_index is not None else raw_sources
    if group.op == "fuse_moe":
      fused = jnp.concatenate(sources, axis=-1)
      return [(key, fused[i]) for i, key in enumerate(group.targets)]
    if group.op == "raise":
      raise ValueError("conversion failed")
    return [(key, sources[0][i]) for i, key in enumerate(group.targets)]

  def convert(self, src_pytree, target_state):
    """Runs `_build_plan` and `_execute_group` over flattened state dicts."""
    src_flat = traverse_util.flatten_dict(src_pytree.to_pure_dict())
    tgt_flat = traverse_util.flatten_dict(target_state.to_pure_dict())
    result = {}
    for group in self._build_plan(src_flat, tgt_flat, frozenset()):
      result.update(self._execute_group(group, dict(src_flat), tgt_flat))
    nested = traverse_util.unflatten_dict(result)
    return {"model": jax.tree_util.tree_map(nnx.Param, nested)}


class _ScannedLayers(nnx.Module):
  """Synthetic scanned MoE layer parameters."""

  def __init__(self, prefused: bool, shape_prefix: tuple[int, ...] = (2,)):
    n = 2 if shape_prefix == (2,) else shape_prefix[0] * shape_prefix[1]
    if prefused:
      self.wi = nnx.Param(jnp.arange(float(n * 32)).reshape(*shape_prefix, 4, 8))
    elif shape_prefix == (2,):
      self.wi_0 = nnx.Param(jnp.full((2, 4, 4), 1.0))
      self.wi_1 = nnx.Param(jnp.full((2, 4, 4), 2.0))
    else:
      self.wi_0 = nnx.Param(jnp.arange(float(n * 16)).reshape(*shape_prefix, 4, 4))
      self.wi_1 = nnx.Param((100.0 + jnp.arange(float(n * 16))).reshape(*shape_prefix, 4, 4))
    self.wo = nnx.Param(jnp.arange(float(n * 16)).reshape(*shape_prefix, 4, 4))


class _InhomogeneousLayers(nnx.Module):
  """Synthetic nested `local_layers` container with cycle slots."""

  def __init__(self, prefused: bool, num_slots: int = 2):
    self.local_layers = _ScannedLayers(prefused, shape_prefix=(2, num_slots))


class _FlatCycleScannedLayers(nnx.Module):
  """Synthetic flat `layer_0..layer_{n-1}` cycle container."""

  def __init__(self, prefused: bool, num_slots: int = 2):
    for slot in range(num_slots):
      setattr(self, f"layer_{slot}", _ScannedLayers(prefused))


class _ScannedModel(nnx.Module):
  """Synthetic scanned MaxText model with non-Param state alongside Params."""

  def __init__(self, prefused: bool, num_slots: int | None = None, flat_cycle: bool = False):
    if num_slots is None:
      self.layers = _ScannedLayers(prefused)
    elif flat_cycle:
      self.layers = _FlatCycleScannedLayers(prefused, num_slots)
    else:
      self.layers = _InhomogeneousLayers(prefused, num_slots)
    self.cache = nnx.Cache(jnp.zeros((2,)))


class _FakeTarget:
  """In-process vLLM target stub with `nnx.Param` weights and one `nnx.BatchStat` leaf."""

  def __init__(self, free_on_update: bool = False, num_layers: int = 2):
    self.state = nnx.State(
        {
            **{
                f"layers_{i}": {
                    "wi": nnx.Param(jnp.zeros((4, 8))),
                    "wo": nnx.Param(jnp.zeros((4, 4))),
                }
                for i in range(num_layers)
            },
            "stat": nnx.BatchStat(jnp.zeros((3,))),
        }
    )
    self.free_on_update = free_on_update
    self.calls = []

  @property
  def transformer_state(self) -> nnx.State:
    """Returns the rollout's NNX state."""
    return self.state

  def delete_cache(self) -> None:
    """Records KV cache deletion."""
    self.calls.append("delete_cache")

  def update_params(self, updated_weights) -> None:
    """Updates rollout weights while preserving non-Param state."""
    self.calls.append("update_params")
    self.state = nnx.State({**updated_weights, "stat": self.state["stat"]})
    if self.free_on_update:
      jax.tree_util.tree_leaves(updated_weights["layers_0"]["wo"])[0].delete()


class LoadAndConvertScannedCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _FakeConverter.instances = []

  def _load(self, model, target, converter_cls=_FakeConverter, **kwargs):
    """Invokes `load_and_convert_scanned_checkpoint` with mocked config/restore."""
    args = {
        "mesh_tp": 2,
        "ckpt_prefuse_moe": False,
        "fp8_moe": False,
        "maxtext_config_overrides": {"model_name": "qwen3-test"},
    }
    args.update(kwargs)
    with (
        mock.patch.object(
            scanned_checkpoint_loader.pyconfig,
            "initialize",
            return_value=mock.sentinel.scanned_cfg,
        ) as init_cfg,
        mock.patch.object(
            scanned_checkpoint_loader.model_creation_utils,
            "from_pretrained",
            return_value=(model, mock.sentinel.mesh),
        ) as from_pretrained,
        mock.patch.object(
            scanned_checkpoint_loader,
            "MaxTextToMaxTextConverter",
            converter_cls,
        ),
    ):
      scanned_checkpoint_loader.load_and_convert_scanned_checkpoint("/ckpt/items", target, **args)
    return init_cfg, from_pretrained

  def test_loads_scanned_checkpoint_and_frees_sources(self):
    model = _ScannedModel(prefused=False)
    model_pure = nnx.state(model, nnx.Param).to_pure_dict()
    scanned_wo = model_pure["layers"]["wo"]
    scanned_wi_0 = model_pure["layers"]["wi_0"]
    scanned_wi_1 = model_pure["layers"]["wi_1"]
    target = _FakeTarget()
    self.assertIsInstance(target, scanned_checkpoint_loader.ScannedCheckpointTarget)
    placeholder = target.state.to_pure_dict()["layers_0"]["wo"]
    stat = target.state.to_pure_dict()["stat"]

    with mock.patch.object(jax, "clear_caches") as clear_caches:
      init_cfg, from_pretrained = self._load(model, target)

    clear_caches.assert_not_called()
    init_args, init_kwargs = init_cfg.call_args
    self.assertEqual(init_args, (["", os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")],))
    self.assertEqual(
        init_kwargs,
        {
            "model_name": "qwen3-test",
            "load_parameters_path": "/ckpt/items",
            "scan_layers": True,
            "prefuse_moe_weights": False,
            "attention": "dot_product",
            "model_call_mode": "",
        },
    )
    from_pretrained.assert_called_once_with(
        mock.sentinel.scanned_cfg,
        devices=jax.devices(),
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertEqual(
        converter.kwargs,
        {
            "config": mock.sentinel.scanned_cfg,
            "tp": 2,
            "prefuse_moe_weights": True,
            "target_dtype": None,
            "fp8_moe": False,
        },
    )
    self.assertEqual(converter.planned_sources, [("layers", "wi_0"), ("layers", "wi_1"), ("layers", "wo")])
    self.assertEqual(target.calls, ["delete_cache", "update_params"])
    self.assertTrue(placeholder.is_deleted())
    self.assertTrue(scanned_wo.is_deleted())
    self.assertTrue(scanned_wi_0.is_deleted())
    self.assertTrue(scanned_wi_1.is_deleted())
    self.assertFalse(stat.is_deleted())
    loaded = target.state.to_pure_dict()
    self.assertEqual(loaded["layers_1"]["wo"][0, 0], 16.0)
    self.assertEqual(loaded["layers_0"]["wi"][0, 7], 2.0)

  def test_warns_that_it_is_deprecated(self):
    # `unittest.assertWarnsRegex` walks `sys.modules.values()` to clear `__warningregistry__`,
    # triggering lazy `transformers` vision imports (`cv2` -> `libxcb.so.1`) in headless CI images.
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always", DeprecationWarning)
      self._load(_ScannedModel(prefused=False), _FakeTarget())
    self.assertTrue(
        any(
            issubclass(w.category, DeprecationWarning) and "use load_scanned_checkpoint" in str(w.message) for w in caught
        ),
        caught,
    )

  def test_custom_base_config_path_is_forwarded(self):
    init_cfg, _ = self._load(_ScannedModel(prefused=False), _FakeTarget(), base_config_path="/custom/base.yml")
    init_args, _ = init_cfg.call_args
    self.assertEqual(init_args, (["", "/custom/base.yml"],))

  def test_prefused_moe_wi_stays_alive_until_its_group_ran(self):
    model = _ScannedModel(prefused=True)
    prefused_wi = nnx.state(model, nnx.Param).to_pure_dict()["layers"]["wi"]
    target = _FakeTarget()
    with mock.patch.object(_FakeConverter, "watched", prefused_wi):
      init_cfg, _ = self._load(model, target, ckpt_prefuse_moe=True, mesh_tp=2)

    _, init_kwargs = init_cfg.call_args
    self.assertTrue(init_kwargs["prefuse_moe_weights"])
    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertNotIn(("layers", "wi"), converter.planned_sources)
    self.assertIn(("layers", "wi_0"), converter.planned_sources)
    self.assertIn(("layers", "wi_1"), converter.planned_sources)
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertTrue(prefused_wi.is_deleted())
    self.assertLen(converter.fuse_inputs, 1)
    for sliced_half in converter.fuse_inputs[0]:
      self.assertTrue(sliced_half.is_deleted())
    loaded = target.state.to_pure_dict()
    # Halves [0:4] and [4:8] of layer 1 are re-fused in order.
    self.assertEqual(loaded["layers_1"]["wi"][0, 0], 32.0)
    self.assertEqual(loaded["layers_1"]["wi"][0, 4], 36.0)

  def test_prefused_moe_wi_with_tp_one_skips_split(self):
    model = _ScannedModel(prefused=True)
    prefused_wi = nnx.state(model, nnx.Param).to_pure_dict()["layers"]["wi"]
    target = _FakeTarget()
    slice_wi_group = _Group("slice", (("layers", "wi"),), (("layers_0", "wi"), ("layers_1", "wi")))
    with mock.patch.object(_FakeConverter, "groups", (_WO_GROUP, slice_wi_group)):
      self._load(model, target, ckpt_prefuse_moe=True, mesh_tp=1)

    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertIn(("layers", "wi"), converter.planned_sources)
    self.assertTrue(prefused_wi.is_deleted())

  def test_shared_local_layers_sources_stay_alive_across_local_indices(self):
    model = _ScannedModel(prefused=False, num_slots=2)
    local = nnx.state(model, nnx.Param).to_pure_dict()["layers"]["local_layers"]
    scanned_wo = local["wo"]
    scanned_wi_0, scanned_wi_1 = local["wi_0"], local["wi_1"]
    expected_wo = jax.device_get(scanned_wo)
    expected_wi = np.concatenate([jax.device_get(scanned_wi_0), jax.device_get(scanned_wi_1)], axis=-1)
    target = _FakeTarget(num_layers=4)

    with (
        mock.patch.object(_FakeConverter, "groups", _SHARED_LOCAL_GROUPS),
        mock.patch.object(_FakeConverter, "watched", scanned_wi_0),
    ):
      self._load(model, target, ckpt_prefuse_moe=False)

    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertTrue(scanned_wo.is_deleted())
    self.assertTrue(scanned_wi_0.is_deleted())
    self.assertTrue(scanned_wi_1.is_deleted())
    loaded = target.state.to_pure_dict()
    for block in range(2):
      for slot in range(2):
        layer = loaded[f"layers_{block * 2 + slot}"]
        np.testing.assert_array_equal(layer["wo"], expected_wo[block, slot])
        np.testing.assert_array_equal(layer["wi"], expected_wi[block, slot])

  def test_prefused_moe_wi_shared_across_local_indices(self):
    model = _ScannedModel(prefused=True, num_slots=2)
    local = nnx.state(model, nnx.Param).to_pure_dict()["layers"]["local_layers"]
    prefused_wi, scanned_wo = local["wi"], local["wo"]
    expected_wi = jax.device_get(prefused_wi)
    expected_wo = jax.device_get(scanned_wo)
    target = _FakeTarget(num_layers=4)

    with (
        mock.patch.object(_FakeConverter, "groups", _SHARED_LOCAL_GROUPS),
        mock.patch.object(_FakeConverter, "watched", prefused_wi),
    ):
      self._load(model, target, ckpt_prefuse_moe=True, mesh_tp=2)

    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertNotIn(("layers", "local_layers", "wi"), converter.planned_sources)
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertLen(converter.fuse_inputs, 2)
    self.assertIs(converter.fuse_inputs[0][0], converter.fuse_inputs[1][0])
    self.assertIs(converter.fuse_inputs[0][1], converter.fuse_inputs[1][1])
    self.assertTrue(prefused_wi.is_deleted())
    self.assertTrue(scanned_wo.is_deleted())
    for sliced_pair in converter.fuse_inputs:
      for sliced_half in sliced_pair:
        self.assertTrue(sliced_half.is_deleted())
    loaded = target.state.to_pure_dict()
    for block in range(2):
      for slot in range(2):
        layer = loaded[f"layers_{block * 2 + slot}"]
        np.testing.assert_array_equal(layer["wo"], expected_wo[block, slot])
        np.testing.assert_array_equal(layer["wi"], expected_wi[block, slot])

  def test_flat_cycle_layers_free_each_slot_before_next_slot_runs(self):
    model = _ScannedModel(prefused=True, num_slots=2, flat_cycle=True)
    pure_layers = nnx.state(model, nnx.Param).to_pure_dict()["layers"]
    slot0_wi, slot0_wo = pure_layers["layer_0"]["wi"], pure_layers["layer_0"]["wo"]
    slot1_wi, slot1_wo = pure_layers["layer_1"]["wi"], pure_layers["layer_1"]["wo"]
    target = _FakeTarget(num_layers=4)
    flat_groups = (
        _Group("slice", (("layers", "layer_0", "wo"),), (("layers_0", "wo"), ("layers_2", "wo"))),
        _Group(
            "fuse_moe",
            (("layers", "layer_0", "wi_0"), ("layers", "layer_0", "wi_1")),
            (("layers_0", "wi"), ("layers_2", "wi")),
        ),
        _Group("slice", (("layers", "layer_1", "wo"),), (("layers_1", "wo"), ("layers_3", "wo"))),
        _Group(
            "fuse_moe",
            (("layers", "layer_1", "wi_0"), ("layers", "layer_1", "wi_1")),
            (("layers_1", "wi"), ("layers_3", "wi")),
        ),
    )

    with (
        mock.patch.object(_FakeConverter, "groups", flat_groups),
        mock.patch.object(_FakeConverter, "watched", slot0_wi),
    ):
      self._load(model, target, ckpt_prefuse_moe=True, mesh_tp=2)

    converter = _FakeConverter.instances[0]
    # `slot0_wi` is alive during slot 0's `fuse_moe`, then freed before slot 1's `fuse_moe`.
    self.assertFalse(converter.watched_alive_during_fuse)
    for arr in (slot0_wi, slot0_wo, slot1_wi, slot1_wo):
      self.assertTrue(arr.is_deleted())

  def test_fp8_moe_reaches_the_converter(self):
    self._load(_ScannedModel(prefused=False), _FakeTarget(), fp8_moe=True)
    self.assertTrue(_FakeConverter.instances[0].kwargs["fp8_moe"])

  def test_reinitialize_cache_target_updates_state_in_place_without_update_params(self):
    class _SamplerTarget(_FakeTarget):
      """Mimics `VllmSampler`/`InprocessVllmSampler` exposing `reinitialize_cache`."""

      def __init__(self):
        super().__init__()
        self.weights_ready_at_reinit = False

      def update_params(self, updated_weights) -> None:
        raise AssertionError("update_params should not be called when reinitialize_cache is available")

      def reinitialize_cache(self) -> None:
        self.calls.append("reinitialize_cache")
        pure = self.state.to_pure_dict()
        self.weights_ready_at_reinit = float(pure["layers_1"]["wo"][0, 0]) == 16.0

    target = _SamplerTarget()
    placeholder = target.state.to_pure_dict()["layers_0"]["wo"]
    stat = target.state.to_pure_dict()["stat"]

    self._load(_ScannedModel(prefused=False), target)

    self.assertEqual(target.calls, ["delete_cache", "reinitialize_cache"])
    self.assertTrue(target.weights_ready_at_reinit)
    self.assertTrue(placeholder.is_deleted())
    self.assertFalse(stat.is_deleted())
    loaded = target.state.to_pure_dict()
    self.assertEqual(loaded["layers_1"]["wo"][0, 0], 16.0)
    self.assertEqual(loaded["layers_0"]["wi"][0, 7], 2.0)

  def test_unstack_local_layers_restores_get_nnx_create_model_fn_on_success_and_error(self):
    def orig_get_create_fn(*args, **kwargs):
      del args, kwargs
      return lambda: _ScannedModel(prefused=False)

    seen_during_restore = []

    def fake_from_pretrained(cfg, **kwargs):
      del cfg, kwargs
      current = scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn
      seen_during_restore.append(current)
      return current()(), mock.sentinel.mesh

    with (
        mock.patch.object(
            scanned_checkpoint_loader.model_creation_utils,
            "get_nnx_create_model_fn",
            orig_get_create_fn,
            create=True,
        ),
        mock.patch.object(scanned_checkpoint_loader.pyconfig, "initialize", return_value=mock.sentinel.scanned_cfg),
        mock.patch.object(scanned_checkpoint_loader, "MaxTextToMaxTextConverter", _FakeConverter),
    ):
      with mock.patch.object(
          scanned_checkpoint_loader.model_creation_utils, "from_pretrained", side_effect=fake_from_pretrained
      ):
        scanned_checkpoint_loader._load(  # pylint: disable=protected-access
            "/ckpt/items",
            _FakeTarget(),
            config_argv=["", "base.yml"],
            restore_overrides={},
            split_keys=lambda s, t, p: (),
            tp=1,
            fp8_moe=False,
            unstack_local_layers=True,
        )
      self.assertLen(seen_during_restore, 1)
      self.assertIsNot(seen_during_restore[0], orig_get_create_fn)
      self.assertIs(scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn, orig_get_create_fn)

      def failing_from_pretrained(cfg, **kwargs):
        del cfg, kwargs
        self.assertIsNot(scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn, orig_get_create_fn)
        raise RuntimeError("restore boom")

      with mock.patch.object(
          scanned_checkpoint_loader.model_creation_utils, "from_pretrained", side_effect=failing_from_pretrained
      ):
        with self.assertRaisesRegex(RuntimeError, "restore boom"):
          scanned_checkpoint_loader._load(  # pylint: disable=protected-access
              "/ckpt/items",
              _FakeTarget(),
              config_argv=["", "base.yml"],
              restore_overrides={},
              split_keys=lambda s, t, p: (),
              tp=1,
              fp8_moe=False,
              unstack_local_layers=True,
          )
      self.assertIs(scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn, orig_get_create_fn)

  def test_freed_vllm_weight_fails_the_load(self):
    target = _FakeTarget(free_on_update=True)
    with self.assertRaisesRegex(RuntimeError, "1 vLLM weight"):
      self._load(_ScannedModel(prefused=False), target)

  def test_converter_hooks_are_restored_when_conversion_fails(self):
    target = _FakeTarget()
    with mock.patch.object(
        _FakeConverter,
        "groups",
        (_Group("raise", (("layers", "wo"),), (("layers_0", "wo"),)),),
    ):
      with self.assertRaisesRegex(ValueError, "conversion failed"):
        self._load(_ScannedModel(prefused=True), target, ckpt_prefuse_moe=True)
    self.assertLen(_FakeConverter.instances, 1)
    converter = _FakeConverter.instances[0]
    self.assertIs(
        converter._execute_group.__func__,  # pylint: disable=protected-access
        _FakeConverter._execute_group,  # pylint: disable=protected-access
    )
    self.assertIs(
        converter._build_plan.__func__,  # pylint: disable=protected-access
        _FakeConverter._build_plan,  # pylint: disable=protected-access
    )
    self.assertEqual(target.calls, ["delete_cache"])


class PrefusedWiKeysToSplitTest(absltest.TestCase):
  """`_prefused_wi_keys_to_split` fails loudly instead of silently skipping the re-layout."""

  _WI = ("decoder", "layers", "MoeBlock_0", "wi")

  def _split(self, src_flat, plan):
    return scanned_checkpoint_loader._prefused_wi_keys_to_split(  # pylint: disable=protected-access
        (self._WI,), src_flat, {("decoder", "layers_0", "MoeBlock_0", "wi"): jnp.zeros((2, 4, 8))}, lambda s, t: plan
    )

  def test_prefused_key_missing_from_source_raises(self):
    with self.assertRaisesRegex(RuntimeError, r"decoder\.layers\.MoeBlock_0\.wi .*not in the restored scanned state"):
      self._split({("base",) + self._WI: jnp.zeros((2, 2, 4, 8))}, plan=[])

  def test_prefused_key_feeding_no_target_raises(self):
    unrelated = pytypes.SimpleNamespace(
        target_key=("decoder", "layers_0", "MoeBlock_0", "wi"), source_keys=(("decoder", "layers", "other", "wi"),)
    )
    with self.assertRaisesRegex(RuntimeError, r"decoder\.layers\.MoeBlock_0\.wi .*feeds no rollout parameter"):
      self._split({self._WI: jnp.zeros((2, 2, 4, 8))}, plan=[unrelated])


class _TreeModule(nnx.Module):
  """Wraps a nested dict of arrays into an `nnx.Module` tree of `nnx.Param`s."""

  def __init__(self, tree: dict):
    for key, value in tree.items():
      setattr(self, key, _TreeModule(value) if isinstance(value, dict) else nnx.Param(value))


class _RolloutModelTarget:
  """Realistic vLLM rollout target wrapping a `MaxTextForCausalLM`-shaped `model` tree."""

  def __init__(self, model_params: dict, kv_cache: jax.Array):
    param_tree = jax.tree_util.tree_map(nnx.Param, model_params)
    self.state = nnx.State({"model": {**param_tree, "kv_cache": nnx.Cache(kv_cache)}})
    self.calls = []

  @property
  def transformer_state(self) -> nnx.State:
    """Returns the rollout's NNX state."""
    return self.state

  def delete_cache(self) -> None:
    """Records KV cache deletion."""
    self.calls.append("delete_cache")

  def update_params(self, updated_weights) -> None:
    """Updates rollout `model` weights while preserving `kv_cache`."""
    self.calls.append("update_params")
    self.state = nnx.State({"model": {**updated_weights, "kv_cache": self.state["model"]["kv_cache"]}})


class _RolloutSamplerTarget(_RolloutModelTarget):
  """`VllmSampler`/`InprocessVllmSampler`-shaped target exposing `reinitialize_cache`."""

  def update_params(self, updated_weights) -> None:
    raise AssertionError("update_params should not be called when reinitialize_cache is available")

  def reinitialize_cache(self) -> None:
    self.calls.append("reinitialize_cache")


def _ramp(*shape: int, offset: float = 0.0) -> jax.Array:
  """Returns a distinct float32 ramp array of `shape` shifted by `offset`."""
  size = int(np.prod(shape))
  return jnp.asarray((np.arange(size, dtype=np.float32) + offset).reshape(shape))


class ScannedCheckpointLoaderIntegrationTest(absltest.TestCase):
  """End-to-end integration tests with the real `MaxTextToMaxTextConverter`."""

  def test_converts_inhomogeneous_local_layers_and_prefused_moe_end_to_end(self):
    # Hybrid 6-layer model with cycle=3 -> 2 scanned blocks:
    # slots 0 and 1 live in nested `local_layers` (axis `scan_axis + 1 = 2`),
    # slot 2 lives in `global_layer` (`scan_axis = 1`).
    cycle = 3
    num_layers = 6
    num_blocks = num_layers // cycle
    scan_axis = 1
    emb = 4
    experts = 2
    moe_dim = 4
    tp = 2

    cfg = pytypes.SimpleNamespace(
        num_decoder_layers=num_layers,
        inhomogeneous_layer_cycle_interval=cycle,
        param_scan_axis=scan_axis,
        dtype="float32",
        weight_dtype="float32",
        fp8_moe=False,
        rollout_fp8_moe=False,
        kv_tp_size=0,
        moe_mlp_tp_size=0,
    )

    use_named_sharding = jax.device_count() >= tp
    if use_named_sharding:
      num_mesh_devices = (jax.device_count() // tp) * tp
      mesh_devices = np.asarray(jax.devices()[:num_mesh_devices])
      src_mesh = jax.sharding.Mesh(mesh_devices, ("fsdp",))
      src_replicated = jax.sharding.NamedSharding(src_mesh, jax.sharding.PartitionSpec())
      tgt_mesh = jax.sharding.Mesh(mesh_devices.reshape(-1, tp), ("dp", "tensor"))
      replicated_sharding = jax.sharding.NamedSharding(tgt_mesh, jax.sharding.PartitionSpec())
      wi_sharding = jax.sharding.NamedSharding(tgt_mesh, jax.sharding.PartitionSpec(None, None, "tensor"))
      wo_sharding = jax.sharding.NamedSharding(tgt_mesh, jax.sharding.PartitionSpec(None, "tensor", None))
    else:
      src_replicated = None
      replicated_sharding = None
      wi_sharding = None
      wo_sharding = None

    def _place(arr: jax.Array, sharding) -> jax.Array:
      return jax.device_put(arr, sharding) if sharding is not None else arr

    for ckpt_prefuse_moe in (False, True):
      with self.subTest(ckpt_prefuse_moe=ckpt_prefuse_moe):
        # Build distinct per-slot scanned tensors so slot/block mixups fail loudly.
        per_slot = {}
        raw_gate_up = {}
        for slot in range(cycle):
          base = float((slot + 1) * 1000)
          wi_0 = _ramp(experts, num_blocks, emb, moe_dim, offset=base + 100.0)
          wi_1 = _ramp(experts, num_blocks, emb, moe_dim, offset=base + 500.0)
          raw_gate_up[slot] = (np.asarray(wi_0), np.asarray(wi_1))
          moe_block = {
              "gate": {"kernel": _ramp(emb, num_blocks, experts, offset=base + 50.0)},
              "wo": _ramp(experts, num_blocks, moe_dim, emb, offset=base + 900.0),
          }
          if ckpt_prefuse_moe:
            # Trainer `prefuse_moe_weights=True` concatenates `[gate | up]` on `-1`.
            moe_block["wi"] = jnp.concatenate([wi_0, wi_1], axis=-1)
          else:
            moe_block["wi_0"] = wi_0
            moe_block["wi_1"] = wi_1
          layer_dict = {
              "input_layernorm": {"scale": _ramp(emb, num_blocks, offset=base)},
              "moe_block": moe_block,
          }
          if slot == cycle - 1:
            layer_dict["self_attention"] = {"query": {"kernel": _ramp(emb, num_blocks, 2, 2, offset=base + 30.0)}}
          else:
            layer_dict["linear_attention"] = {"conv": {"kernel": _ramp(emb, num_blocks, 1, 3, offset=base + 30.0)}}
          per_slot[slot] = layer_dict

        local_layers = jax.tree.map(
            lambda *slots: _place(jnp.stack(slots, axis=scan_axis + 1), src_replicated),
            *(per_slot[s] for s in range(cycle - 1)),
        )
        global_layer = jax.tree.map(lambda arr: _place(arr, src_replicated), per_slot[cycle - 1])

        # `token_embedder` shares `replicated_sharding` directly (`out.sharding == sharding`),
        # while `decoder_norm` sits on `src_replicated` (`out.sharding.is_equivalent_to(sharding)`).
        embed_arr = _place(_ramp(8, emb, offset=10.0), replicated_sharding)
        norm_arr = _place(_ramp(emb, offset=20.0), src_replicated)
        expected_embed = np.asarray(embed_arr)
        expected_norm = np.asarray(norm_arr)
        expected_slots = jax.tree.map(np.asarray, per_slot)

        source_tree = {
            "base": {
                "token_embedder": {"embedding": embed_arr},
                "decoder": {
                    "decoder_norm": {"scale": norm_arr},
                    "layers": {
                        "local_layers": local_layers,
                        "global_layer": global_layer,
                    },
                },
            }
        }
        scanned_model = _TreeModule(source_tree)
        scanned_layer_leaves = jax.tree_util.tree_leaves(
            nnx.state(scanned_model.base.decoder.layers, nnx.Param).to_pure_dict()
        )

        # Build unrolled rollout target (`layers_0`..`layers_5`) with pre-fused `wi`.
        tgt_decoder = {"decoder_norm": {"scale": _place(jnp.zeros((emb,), dtype=jnp.float32), replicated_sharding)}}
        for layer_idx in range(num_layers):
          slot = layer_idx % cycle
          tgt_block = {
              "input_layernorm": {
                  "scale": _place(jnp.zeros((emb,), dtype=jnp.float32), replicated_sharding),
              },
              "moe_block": {
                  "gate": {
                      "kernel": _place(jnp.zeros((emb, experts), dtype=jnp.float32), replicated_sharding),
                  },
                  "wi": _place(jnp.zeros((experts, emb, moe_dim * 2), dtype=jnp.float32), wi_sharding),
                  "wo": _place(jnp.zeros((experts, moe_dim, emb), dtype=jnp.float32), wo_sharding),
              },
          }
          if slot == cycle - 1:
            tgt_block["self_attention"] = {
                "query": {"kernel": _place(jnp.zeros((emb, 2, 2), dtype=jnp.float32), replicated_sharding)}
            }
          else:
            tgt_block["linear_attention"] = {
                "conv": {"kernel": _place(jnp.zeros((emb, 1, 3), dtype=jnp.float32), replicated_sharding)}
            }
          tgt_decoder[f"layers_{layer_idx}"] = tgt_block

        tgt_model_params = {
            "token_embedder": {
                "embedding": _place(jnp.zeros((8, emb), dtype=jnp.float32), replicated_sharding),
            },
            "decoder": tgt_decoder,
        }
        placeholder_leaves = jax.tree_util.tree_leaves(tgt_model_params)
        kv_cache = jnp.ones((2, 4), dtype=jnp.float32)
        target = _RolloutModelTarget(tgt_model_params, kv_cache)
        self.assertIsInstance(target, scanned_checkpoint_loader.ScannedCheckpointTarget)

        with (
            mock.patch.object(scanned_checkpoint_loader.pyconfig, "initialize", return_value=cfg),
            mock.patch.object(
                scanned_checkpoint_loader.model_creation_utils,
                "from_pretrained",
                return_value=(scanned_model, mock.sentinel.mesh),
            ),
        ):
          scanned_checkpoint_loader.load_and_convert_scanned_checkpoint(
              "/ckpt/hybrid",
              target,
              mesh_tp=tp,
              ckpt_prefuse_moe=ckpt_prefuse_moe,
              fp8_moe=False,
              maxtext_config_overrides={"model_name": "qwen3_5-test"},
          )

        self.assertEqual(target.calls, ["delete_cache", "update_params"])
        self.assertFalse(kv_cache.is_deleted())
        for placeholder in placeholder_leaves:
          self.assertTrue(placeholder.is_deleted())
        for scanned_leaf in scanned_layer_leaves:
          self.assertTrue(scanned_leaf.is_deleted())

        loaded = target.transformer_state.to_pure_dict()["model"]
        np.testing.assert_array_equal(np.asarray(loaded["token_embedder"]["embedding"]), expected_embed)
        np.testing.assert_array_equal(np.asarray(loaded["decoder"]["decoder_norm"]["scale"]), expected_norm)
        if use_named_sharding:
          self.assertEqual(loaded["decoder"]["decoder_norm"]["scale"].sharding, replicated_sharding)

        shard_width = moe_dim // tp
        for layer_idx in range(num_layers):
          slot, block = layer_idx % cycle, layer_idx // cycle
          got_layer = loaded["decoder"][f"layers_{layer_idx}"]
          want_slot = expected_slots[slot]

          np.testing.assert_array_equal(
              np.asarray(got_layer["input_layernorm"]["scale"]),
              want_slot["input_layernorm"]["scale"][:, block],
          )
          np.testing.assert_array_equal(
              np.asarray(got_layer["moe_block"]["gate"]["kernel"]),
              want_slot["moe_block"]["gate"]["kernel"][:, block, :],
          )
          np.testing.assert_array_equal(
              np.asarray(got_layer["moe_block"]["wo"]),
              want_slot["moe_block"]["wo"][:, block, :, :],
          )
          if slot == cycle - 1:
            np.testing.assert_array_equal(
                np.asarray(got_layer["self_attention"]["query"]["kernel"]),
                want_slot["self_attention"]["query"]["kernel"][:, block, :, :],
            )
          else:
            np.testing.assert_array_equal(
                np.asarray(got_layer["linear_attention"]["conv"]["kernel"]),
                want_slot["linear_attention"]["conv"]["kernel"][:, block, :, :],
            )

          # Verify TP=2 per-shard `[local_gate, local_up]` interleaving on `wi`.
          w0_layer = raw_gate_up[slot][0][:, block, :, :]
          w1_layer = raw_gate_up[slot][1][:, block, :, :]
          expected_wi = np.concatenate(
              [
                  w0_layer[:, :, :shard_width],
                  w1_layer[:, :, :shard_width],
                  w0_layer[:, :, shard_width:],
                  w1_layer[:, :, shard_width:],
              ],
              axis=-1,
          )
          plain_concat_wi = np.concatenate([w0_layer, w1_layer], axis=-1)
          got_wi = got_layer["moe_block"]["wi"]
          if use_named_sharding:
            self.assertEqual(got_wi.sharding, wi_sharding)
          self.assertFalse(np.array_equal(expected_wi, plain_concat_wi))
          np.testing.assert_array_equal(np.asarray(got_wi), expected_wi)


def _tiny_config(**overrides) -> dict:
  """Returns MaxText overrides for a tiny 2-layer model that runs on 4 CPU devices."""
  return {
      "decoder_block": "mixtral",
      "base_num_decoder_layers": 2,
      "base_emb_dim": 16,
      "base_mlp_dim": 32,
      "base_moe_mlp_dim": 32,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 8,
      "vocab_size": 64,
      "max_target_length": 16,
      "max_prefill_predict_length": 8,
      "per_device_batch_size": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "skip_jax_distributed_system": True,
      "enable_checkpointing": False,
      "sparse_matmul": False,
      "megablox": False,
      **overrides,
  }


_TINY_MOE = _tiny_config(num_experts=4, num_experts_per_tok=2)
_TINY_QWEN3_5_MOE = _tiny_config(
    decoder_block="qwen3_5",
    base_num_decoder_layers=8,
    inhomogeneous_layer_cycle_interval=4,
    num_experts=4,
    shared_experts=1,
    num_experts_per_tok=2,
    gdn_num_key_heads=2,
    gdn_num_value_heads=2,
    gdn_key_head_dim=8,
    gdn_value_head_dim=8,
    gdn_conv_kernel_dim=4,
    gdn_chunk_size=8,
)
_TINY_DENSE_RELU = _tiny_config(decoder_block="default", mlp_activations=["relu"])
_TINY_DENSE_FUSED = _tiny_config(decoder_block="default", fused_mlp=True)
_MOE_PATH = ("decoder", "layers", "MoeBlock_0")
_QWEN3_5_NESTED_MOE_PATHS = (
    ("decoder", "layers", "global_layer", "mlp", "routed_experts"),
    ("decoder", "layers", "local_layers", "mlp", "routed_experts"),
)
_QWEN3_5_FLAT_MOE_PATHS = tuple(("decoder", "layers", f"layer_{i}", "mlp", "routed_experts") for i in range(4))
_REAL_MODEL_CREATION = scanned_checkpoint_loader.model_creation_utils.from_pretrained is not None


def _pyconfig(maxtext_config: dict, **overrides):
  """Initializes a MaxText config from `base.yml` and `maxtext_config`."""
  return scanned_checkpoint_loader.pyconfig.initialize(
      ["", os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")],
      **{**maxtext_config, "attention": "dot_product", "model_call_mode": "", **overrides},
  )


def _random_scanned_params(maxtext_config: dict) -> dict:
  """Returns randomly initialized scanned, unfused params of the model as a pure dict."""
  cfg = _pyconfig(maxtext_config, scan_layers=True, prefuse_moe_weights=False)
  model, _ = scanned_checkpoint_loader.model_creation_utils.from_pretrained(
      cfg, devices=jax.devices(), model_mode=MODEL_MODE_AUTOREGRESSIVE
  )
  return nnx.to_pure_dict(nnx.state(model, nnx.Param))


def _prefuse(params: dict) -> dict:
  """Concatenates routed MoE `[wi_0 | wi_1]` into `wi` on the last axis, as a `prefuse_moe_weights` trainer stores it."""
  flat = traverse_util.flatten_dict(params)
  for key in [k for k in flat if k[-1] == "wi_0" and len(k) >= 2 and k[-2] in ("MoeBlock_0", "routed_experts")]:
    flat[key[:-1] + ("wi",)] = jnp.concatenate([flat.pop(key), flat.pop(key[:-1] + ("wi_1",))], axis=-1)
  return traverse_util.unflatten_dict(flat)


def _to_flat_cycle_params(params: dict, *, cycle: int = 4, scan_axis: int = 1) -> dict:
  """Converts nested `decoder.layers.{local_layers, global_layer}` params into flat `layer_0..layer_{cycle-1}`."""
  layers = params["decoder"]["layers"]
  local_layers = layers["local_layers"]
  flat_layers = {
      f"layer_{slot}": jax.tree.map(
          lambda arr, s=slot: jax.lax.index_in_dim(
              arr, s, axis=(scan_axis + 1 if arr.ndim - 1 > scan_axis else 0), keepdims=False
          ),
          local_layers,
      )
      for slot in range(cycle - 1)
  }
  flat_layers[f"layer_{cycle - 1}"] = layers["global_layer"]
  return {
      **params,
      "decoder": {
          **params["decoder"],
          "layers": flat_layers,
      },
  }


def _rng_leaves(scan_len: int | None = None) -> dict:
  """Builds NNX `RngState`-shaped `{count, key}` leaves Wrapped in `{"value": ...}` as RL trainer checkpoints save."""
  count_shape = () if scan_len is None else (scan_len,)
  key_shape = (2,) if scan_len is None else (scan_len, 2)
  return {
      stream: {
          "count": {"value": jnp.zeros(count_shape, dtype=jnp.uint32)},
          "key": {"value": jnp.ones(key_shape, dtype=jnp.uint32)},
      }
      for stream in ("aqt", "dropout", "params")
  }


def _save(directory: str, params: dict, *, flavor: str, use_ocdbt: bool = True, use_zarr3: bool = True) -> str:
  """Writes `params` as a scanned checkpoint step dir and returns its path.

  Args:
    directory: Step dir to create.
    params: Pure-dict model params.
    flavor: `"linen"` uses `checkpointing.save_params_to_path`, which stores
      `{"params": {"params": ...}}` under `items`. `"nnx"` stores
      `{"value": ...}`-wrapped leaves under `items` as
      `layerwise_quantization` does. `"nnx_rl_model_params"` stores a
      `"base"`-wrapped NNX tree with extra `rngs` leaves under `model_params`,
      matching Tunix/MaxText RL trainer checkpoints.
    use_ocdbt: Whether to store with OCDBT.
    use_zarr3: Whether to store Zarr v3 arrays.

  Returns:
    `directory`.
  """
  from maxtext.common import checkpointing  # pylint: disable=import-outside-toplevel

  if flavor == "linen":
    checkpointing.save_params_to_path(directory, {"params": params}, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  elif flavor == "nnx_rl_model_params":
    wrapped = jax.tree.map(lambda leaf: {"value": leaf}, params)
    wrapped["decoder"] = {**wrapped["decoder"], "rngs": _rng_leaves()}
    layers = dict(wrapped["decoder"]["layers"])
    layers["rngs"] = _rng_leaves(scan_len=2)
    for name, sub in list(layers.items()):
      if name.startswith("layer_") and isinstance(sub, dict) and "mlp" in sub:
        routed = {**sub["mlp"]["routed_experts"], "rngs": _rng_leaves(scan_len=2)}
        layers[name] = {**sub, "mlp": {**sub["mlp"], "routed_experts": routed}}
    wrapped["decoder"]["layers"] = layers
    ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3).save(
        os.path.join(directory, "model_params"), {"base": wrapped}, force=True
    )
  else:
    wrapped = jax.tree.map(lambda leaf: {"value": leaf}, params)
    ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3).save(
        os.path.join(directory, "items"), wrapped, force=True
    )
  return directory


def _rollout_target(
    maxtext_config: dict, *, prefuse: bool, shard_wi: bool, sampler_reinit: bool = False
) -> "_RolloutModelTarget":
  """Builds a zero-filled unscanned rollout target of the model on a 2x2 CPU mesh.

  Args:
    maxtext_config: The model's MaxText overrides.
    prefuse: The rollout's `prefuse_moe_weights`.
    shard_wi: Whether to shard each MoE `wi` 2 ways on its last axis; other
      weights are replicated.
    sampler_reinit: Whether to expose `reinitialize_cache` like `VllmSampler`.

  Returns:
    The target.
  """
  cfg = _pyconfig(maxtext_config, scan_layers=False, prefuse_moe_weights=prefuse)
  _, abstract_model = scanned_checkpoint_loader.model_creation_utils.create_nnx_abstract_model(
      cfg, devices=jax.devices(), model_mode=MODEL_MODE_AUTOREGRESSIVE
  )
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "tensor"))
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  wi_sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None, "tensor"))
  flat = traverse_util.flatten_dict(nnx.to_pure_dict(nnx.state(abstract_model, nnx.Param)))
  params = {
      key: jax.device_put(
          jnp.zeros(leaf.shape, leaf.dtype),
          wi_sharded
          if shard_wi and key[-1] == "wi" and len(key) >= 2 and key[-2] in ("MoeBlock_0", "routed_experts")
          else replicated,
      )
      for key, leaf in flat.items()
  }
  target_cls = _RolloutSamplerTarget if sampler_reinit else _RolloutModelTarget
  return target_cls(traverse_util.unflatten_dict(params), jnp.ones((2, 4), dtype=jnp.float32))


def _expected_weights(maxtext_config: dict, params: dict, target: "_RolloutModelTarget") -> dict:
  """Converts in-memory unfused scanned `params` for `target` with the real converter, without hooks."""
  converter = scanned_checkpoint_loader.MaxTextToMaxTextConverter(
      config=_pyconfig(maxtext_config, scan_layers=True, prefuse_moe_weights=False),
      prefuse_moe_weights=True,
      target_dtype=None,
      fp8_moe=False,
  )
  converted = converter.convert(params, target_state=target.transformer_state)
  return traverse_util.flatten_dict(
      jax.tree.map(lambda param: np.asarray(param[...]), converted, is_leaf=lambda x: isinstance(x, nnx.Variable))
  )


@unittest.skipUnless(_REAL_MODEL_CREATION, "needs the real `model_creation_utils` (TF/Orbax deps)")
class ResolveScannedCheckpointTest(parameterized.TestCase):
  """`resolve_scanned_checkpoint` on checkpoints written by MaxText's writers."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.params = _random_scanned_params(_TINY_MOE)

  @parameterized.product(item_dir=["direct", "items", "default", "model_params"], ocdbt_zarr3=[True, False])
  def test_resolves_param_dir_and_storage(self, item_dir, ocdbt_zarr3):
    step_dir = _save(
        self.create_tempdir().full_path, self.params, flavor="linen", use_ocdbt=ocdbt_zarr3, use_zarr3=ocdbt_zarr3
    )
    params_dir = os.path.join(step_dir, "items")
    if item_dir in ("default", "model_params"):
      os.rename(params_dir, os.path.join(step_dir, item_dir))
      params_dir = os.path.join(step_dir, item_dir)

    storage = scanned_checkpoint_loader.resolve_scanned_checkpoint(params_dir if item_dir == "direct" else step_dir)

    self.assertEqual(
        storage,
        scanned_checkpoint_loader.ScannedCheckpointStorage(
            params_dir=params_dir, use_ocdbt=ocdbt_zarr3, use_zarr3=ocdbt_zarr3
        ),
    )

  def test_does_not_initialize_jax(self):
    step_dir = _save(self.create_tempdir().full_path, self.params, flavor="linen")
    script = (
        "import sys\n"
        "from jax._src import xla_bridge\n"
        "from maxtext.integration.vllm import scanned_checkpoint_loader\n"
        "storage = scanned_checkpoint_loader.resolve_scanned_checkpoint(sys.argv[1])\n"
        "assert storage.use_ocdbt and storage.use_zarr3, storage\n"
        "print(xla_bridge.backends_are_initialized())\n"
    )
    result = subprocess.run([sys.executable, "-c", script, step_dir], capture_output=True, text=True, check=True)
    self.assertEqual(result.stdout.strip().splitlines()[-1], "False")

  def test_missing_metadata_lists_candidates(self):
    root = self.create_tempdir().full_path
    with self.assertRaisesRegex(FileNotFoundError, "items.*default.*model_params"):
      scanned_checkpoint_loader.resolve_scanned_checkpoint(root)

  @parameterized.parameters('["not", "a", "mapping"]', '{"use_zarr3": true}', "{not json")
  def test_unreadable_metadata_raises(self, metadata):
    root = self.create_tempdir()
    root.create_file("items/_METADATA", content=metadata)
    with self.assertRaisesRegex(ValueError, "_METADATA"):
      scanned_checkpoint_loader.resolve_scanned_checkpoint(root.full_path)


_SENTINEL = "SCANNED_CHECKPOINT_LOADER_TESTS_PASSED"
_RAN = re.compile(rf"{_SENTINEL} ran=(\d+)")
_CHILD_TEST_BASE = parameterized.TestCase if _IN_CHILD else object


@unittest.skipUnless(_REAL_MODEL_CREATION, "needs the real `model_creation_utils` (TF/Orbax deps)")
class LoadScannedCheckpointSubprocessTest(absltest.TestCase):
  """Runs `LoadScannedCheckpointTest` in a fresh 4-CPU-device subprocess.

  `cpu-post-training-unit` runs under pytest-xdist with `XLA_FLAGS=""`, and sibling modules
  initialize the CPU backend with 1 device before this file is imported, so `_rollout_target`'s
  2x2 mesh requires a fresh interpreter.
  """

  def test_load_scanned_checkpoint_on_four_cpu_devices(self):
    """Executes `LoadScannedCheckpointTest` in a child interpreter with 4 CPU devices."""
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    env["XLA_FLAGS"] = (f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_REQUIRED_DEVICES}").strip()
    env["JAX_PLATFORMS"] = "cpu"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    if os.path.isdir(src_dir):
      env["PYTHONPATH"] = os.pathsep.join([src_dir, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else src_dir

    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    self.assertEqual(result.returncode, 0, report)
    ran = _RAN.search(result.stdout)
    self.assertIsNotNone(ran, f"child did not report a completed run\n{report}")
    self.assertEqual(int(ran.group(1)), 23, f"unexpected child test count\n{report}")


@unittest.skipUnless(_REAL_MODEL_CREATION, "needs the real `model_creation_utils` (TF/Orbax deps)")
class LoadScannedCheckpointTest(_CHILD_TEST_BASE):
  """`load_scanned_checkpoint` on real checkpoints, with real config, restore and conversion."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.params = _random_scanned_params(_TINY_MOE)
    cls.qwen3_5_params = _random_scanned_params(_TINY_QWEN3_5_MOE)

  def _load(self, step_dir, target, maxtext_config):
    """Loads `step_dir` into `target`, spying on `from_pretrained` and the split keys.

    Returns:
      `(restore_cfg, split_keys, was_unstacked)`: the config `from_pretrained`
      restored with, the pre-fused `wi` keys the loader split, and whether
      `get_nnx_create_model_fn` was wrapped for flat-cycle unstacking during
      `from_pretrained`.
    """
    split_keys = []
    pick_split_keys = scanned_checkpoint_loader._prefused_wi_keys_to_split  # pylint: disable=protected-access
    orig_get_fn = scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn
    orig_from_pretrained = scanned_checkpoint_loader.model_creation_utils.from_pretrained
    was_unstacked = []

    def _spy_split(*args):
      keys = pick_split_keys(*args)
      split_keys.extend(keys)
      return keys

    def _spy_from_pretrained(*args, **kwargs):
      was_unstacked.append(scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn is not orig_get_fn)
      return orig_from_pretrained(*args, **kwargs)

    with (
        mock.patch.object(
            scanned_checkpoint_loader.model_creation_utils,
            "from_pretrained",
            side_effect=_spy_from_pretrained,
        ) as from_pretrained,
        mock.patch.object(scanned_checkpoint_loader, "_prefused_wi_keys_to_split", _spy_split),
    ):
      scanned_checkpoint_loader.load_scanned_checkpoint(step_dir, target, maxtext_config)
    from_pretrained.assert_called_once()
    self.assertIs(scanned_checkpoint_loader.model_creation_utils.get_nnx_create_model_fn, orig_get_fn)
    return from_pretrained.call_args.args[0], split_keys, was_unstacked[0]

  def _assert_loaded(self, target, expected, *, expected_finish_call: str = "update_params"):
    loaded = traverse_util.flatten_dict(nnx.to_pure_dict(target.transformer_state))
    loaded.pop(("model", "kv_cache"))
    self.assertEqual(set(loaded), set(expected))
    for key, want in expected.items():
      np.testing.assert_array_equal(np.asarray(loaded[key]), want, err_msg=".".join(map(str, key)))
    self.assertEqual(target.calls, ["delete_cache", expected_finish_call])

  @parameterized.product(ckpt_prefused=[True, False], flavor=["linen", "nnx"], shard_wi=[True, False])
  def test_loads_moe_checkpoint(self, ckpt_prefused, flavor, shard_wi):
    params = _prefuse(self.params) if ckpt_prefused else self.params
    step_dir = _save(self.create_tempdir().full_path, params, flavor=flavor, use_ocdbt=False, use_zarr3=False)
    target = _rollout_target(_TINY_MOE, prefuse=True, shard_wi=shard_wi)
    wi_key = ("model",) + _MOE_PATH[:1] + ("layers_0",) + _MOE_PATH[2:] + ("wi",)
    wi_sharding = traverse_util.flatten_dict(nnx.to_pure_dict(target.transformer_state))[wi_key].sharding
    expected = _expected_weights(_TINY_MOE, self.params, _rollout_target(_TINY_MOE, prefuse=True, shard_wi=shard_wi))

    restore_cfg, split_keys, was_unstacked = self._load(step_dir, target, {**_TINY_MOE, "prefuse_moe_weights": True})

    self.assertFalse(was_unstacked)
    self.assertEqual(restore_cfg.prefuse_moe_weights, ckpt_prefused)
    self.assertFalse(restore_cfg.checkpoint_storage_use_ocdbt)
    self.assertFalse(restore_cfg.checkpoint_storage_use_zarr3)
    self.assertFalse(restore_cfg.fp8_moe)
    self.assertEqual(restore_cfg.load_parameters_path, os.path.join(step_dir, "items"))
    # Only a pre-fused `wi` feeding a sharded rollout `wi` needs the per-shard re-layout.
    self.assertEqual(split_keys, [_MOE_PATH + ("wi",)] if ckpt_prefused and shard_wi else [])
    self._assert_loaded(target, expected)
    loaded_wi = traverse_util.flatten_dict(nnx.to_pure_dict(target.transformer_state))[wi_key]
    self.assertEqual(loaded_wi.sharding, wi_sharding)

  def test_storage_flags_reach_the_restore(self):
    step_dir = _save(self.create_tempdir().full_path, self.params, flavor="linen", use_ocdbt=True, use_zarr3=True)
    restore_cfg, _, _ = self._load(
        step_dir,
        _rollout_target(_TINY_MOE, prefuse=True, shard_wi=True),
        {**_TINY_MOE, "checkpoint_storage_use_ocdbt": False, "checkpoint_storage_use_zarr3": False},
    )
    self.assertTrue(restore_cfg.checkpoint_storage_use_ocdbt)
    self.assertTrue(restore_cfg.checkpoint_storage_use_zarr3)

  def test_unfused_checkpoint_into_unfused_rollout(self):
    step_dir = _save(self.create_tempdir().full_path, self.params, flavor="linen")
    target = _rollout_target(_TINY_MOE, prefuse=False, shard_wi=False)
    expected = _expected_weights(_TINY_MOE, self.params, _rollout_target(_TINY_MOE, prefuse=False, shard_wi=False))

    restore_cfg, split_keys, _ = self._load(step_dir, target, {**_TINY_MOE, "prefuse_moe_weights": False})

    self.assertFalse(restore_cfg.prefuse_moe_weights)
    self.assertEqual(split_keys, [])
    self._assert_loaded(target, expected)

  def test_prefused_checkpoint_into_unfused_rollout_fails_before_freeing(self):
    step_dir = _save(self.create_tempdir().full_path, _prefuse(self.params), flavor="linen")
    target = _rollout_target(_TINY_MOE, prefuse=False, shard_wi=False)
    with self.assertRaisesRegex(ValueError, "prefuse_moe_weights=False"):
      scanned_checkpoint_loader.load_scanned_checkpoint(step_dir, target, {**_TINY_MOE, "prefuse_moe_weights": False})
    self.assertEqual(target.calls, [])

  @parameterized.named_parameters(("single_activation", _TINY_DENSE_RELU), ("fused_mlp", _TINY_DENSE_FUSED))
  def test_dense_mlp_wi_is_not_moe(self, maxtext_config):
    params = _random_scanned_params(maxtext_config)
    self.assertIn(("decoder", "layers", "mlp", "wi", "kernel"), traverse_util.flatten_dict(params))
    step_dir = _save(self.create_tempdir().full_path, params, flavor="linen")
    target = _rollout_target(maxtext_config, prefuse=True, shard_wi=False)
    expected = _expected_weights(maxtext_config, params, _rollout_target(maxtext_config, prefuse=True, shard_wi=False))

    restore_cfg, split_keys, _ = self._load(step_dir, target, {**maxtext_config, "prefuse_moe_weights": True})

    self.assertFalse(restore_cfg.prefuse_moe_weights)
    self.assertEqual(split_keys, [])
    self._assert_loaded(target, expected)

  def test_moe_model_without_moe_weights_names_the_module(self):
    step_dir = _save(self.create_tempdir().full_path, _random_scanned_params(_TINY_DENSE_RELU), flavor="linen")
    target = _rollout_target(_TINY_MOE, prefuse=True, shard_wi=True)
    with self.assertRaisesRegex(ValueError, "decoder/layers/MoeBlock_0"):
      scanned_checkpoint_loader.load_scanned_checkpoint(step_dir, target, _TINY_MOE)
    self.assertEqual(target.calls, [])

  @parameterized.named_parameters(
      ("nested_unfused_sharded", False, False, "linen", True),
      ("nested_prefused_sharded", False, True, "nnx", True),
      ("flat_linen_unfused_sharded", True, False, "linen", True),
      ("flat_nnx_rl_prefused_ep_unsharded", True, True, "nnx_rl_model_params", False),
      ("flat_nnx_rl_prefused_tp_sharded", True, True, "nnx_rl_model_params", True),
      ("flat_nnx_rl_unfused_ep_unsharded", True, False, "nnx_rl_model_params", False),
  )
  def test_loads_qwen3_5_hybrid_checkpoint(self, flat_cycle, ckpt_prefused, flavor, shard_wi):
    params = _to_flat_cycle_params(self.qwen3_5_params) if flat_cycle else self.qwen3_5_params
    if ckpt_prefused:
      params = _prefuse(params)
    step_dir = _save(
        self.create_tempdir().full_path,
        params,
        flavor=flavor,
        use_ocdbt=False,
        use_zarr3=False,
    )
    use_sampler_reinit = flavor == "nnx_rl_model_params"
    target = _rollout_target(_TINY_QWEN3_5_MOE, prefuse=True, shard_wi=shard_wi, sampler_reinit=use_sampler_reinit)
    expected = _expected_weights(
        _TINY_QWEN3_5_MOE,
        self.qwen3_5_params,
        _rollout_target(_TINY_QWEN3_5_MOE, prefuse=True, shard_wi=shard_wi),
    )
    caller_cfg = {k: v for k, v in _TINY_QWEN3_5_MOE.items() if k != "skip_jax_distributed_system"}
    caller_cfg["prefuse_moe_weights"] = True

    restore_cfg, split_keys, was_unstacked = self._load(step_dir, target, caller_cfg)

    self.assertEqual(was_unstacked, flat_cycle)
    self.assertEqual(restore_cfg.prefuse_moe_weights, ckpt_prefused)
    self.assertTrue(restore_cfg.skip_jax_distributed_system)
    expected_item = "model_params" if flavor == "nnx_rl_model_params" else "items"
    self.assertEqual(restore_cfg.load_parameters_path, os.path.join(step_dir, expected_item))
    moe_paths = _QWEN3_5_FLAT_MOE_PATHS if flat_cycle else _QWEN3_5_NESTED_MOE_PATHS
    want_split = [p + ("wi",) for p in moe_paths] if ckpt_prefused and shard_wi else []
    self.assertEqual(split_keys, want_split)
    self._assert_loaded(
        target,
        expected,
        expected_finish_call="reinitialize_cache" if use_sampler_reinit else "update_params",
    )

  @parameterized.named_parameters(
      ("missing_slot", lambda flat: {k: v for k, v in flat.items() if k != "layer_3"}, r"flat cycle layers.*layer_3"),
      ("extra_slot", lambda flat: {**flat, "layer_4": flat["layer_0"]}, r"flat cycle layers.*layer_4"),
      (
          "mixed_nested_and_flat",
          lambda flat, nested=True: "mixed",
          r"mixes nested.*local_layers.*layer_0",
      ),
  )
  def test_invalid_flat_cycle_checkpoint_fails_before_freeing(self, mutate_fn, err_regex):
    flat_params = _to_flat_cycle_params(self.qwen3_5_params)
    if mutate_fn(flat_params["decoder"]["layers"]) == "mixed":
      bad_layers = {**self.qwen3_5_params["decoder"]["layers"], "layer_0": flat_params["decoder"]["layers"]["layer_0"]}
    else:
      bad_layers = mutate_fn(flat_params["decoder"]["layers"])
    bad_params = {**flat_params, "decoder": {**flat_params["decoder"], "layers": bad_layers}}
    step_dir = _save(self.create_tempdir().full_path, bad_params, flavor="linen", use_ocdbt=False, use_zarr3=False)
    target = _rollout_target(_TINY_QWEN3_5_MOE, prefuse=True, shard_wi=False)
    with self.assertRaisesRegex(ValueError, err_regex):
      scanned_checkpoint_loader.load_scanned_checkpoint(
          step_dir, target, {**_TINY_QWEN3_5_MOE, "prefuse_moe_weights": True}
      )
    self.assertEqual(target.calls, [])


def _child_main(_argv):
  """Runs `LoadScannedCheckpointTest` inside the 4-CPU-device child process."""
  if jax.device_count() < _REQUIRED_DEVICES:
    raise SystemExit(f"expected >= {_REQUIRED_DEVICES} CPU devices, got {jax.device_count()}")
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(LoadScannedCheckpointTest)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  if not result.wasSuccessful():
    sys.exit(1)
  print(f"{_SENTINEL} ran={result.testsRun - len(result.skipped)}")


if __name__ == "__main__":
  if _IN_CHILD:
    app.run(_child_main, argv=[sys.argv[0]])
  else:
    absltest.main()

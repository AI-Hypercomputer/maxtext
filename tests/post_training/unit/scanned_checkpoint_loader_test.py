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
# take effect even when the test module is executed directly via `python`.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import types as pytypes
from typing import NamedTuple
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
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


class _ScannedModel(nnx.Module):
  """Synthetic scanned MaxText model with non-Param state alongside Params."""

  def __init__(self, prefused: bool, num_slots: int | None = None):
    self.layers = _ScannedLayers(prefused) if num_slots is None else _InhomogeneousLayers(prefused, num_slots)
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

  def test_fp8_moe_reaches_the_converter(self):
    self._load(_ScannedModel(prefused=False), _FakeTarget(), fp8_moe=True)
    self.assertTrue(_FakeConverter.instances[0].kwargs["fp8_moe"])

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


if __name__ == "__main__":
  absltest.main()

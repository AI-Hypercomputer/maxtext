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

"""Unit, ZeRO-1 sharding, Orbax checkpoint round-trip, and config tests for optimizer_device_resident_gib."""

import importlib.util
import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from optax.contrib._muon import muon as optax_muon
from orbax.checkpoint import v1 as ocp
from pydantic import ValidationError

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.utils import lora_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


class _TinyBudgetModel(nnx.Module):
  """Compact 2-layer NNX model with distinct parameter sizes for deterministic budget packing tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.embed = nnx.Param(
        jax.random.normal(rngs.params(), (32, 16), dtype=jnp.float32),
        out_sharding=("fsdp", None),
    )
    self.linear1 = nnx.Linear(16, 16, dtype=jnp.float32, rngs=rngs)
    self.linear2 = nnx.Linear(16, 8, dtype=jnp.float32, use_bias=False, rngs=rngs)

  def __call__(self, tokens: jax.Array) -> jax.Array:
    x = self.embed.get_value()[tokens]
    h = jax.nn.silu(self.linear1(x))
    return self.linear2(h)


def _state_config(**overrides):
  """Builds a minimal config stub for get_abstract_state_nnx / setup_initial_state tests."""
  base = {
      "init_weights_seed": 42,
      "shard_optimizer_over_data": False,
      "optimizer_memory_host_offload": False,
      "optimizer_device_resident_gib": 0.0,
      "parameter_memory_host_offload": False,
      "param_scan_axis": 0,
      "logical_axis_rules": [("fsdp", "fsdp"), ("data", "data")],
      "use_m3_model": False,
      "enable_diloco": False,
      "enable_checkpointing": True,
      "enable_continuous_checkpointing": False,
      "enable_emergency_checkpoint": False,
      "enable_multi_tier_checkpointing": False,
      "enable_autocheckpoint": False,
      "save_checkpoint_on_start": True,
      "checkpoint_period": 1,
      "local_checkpoint_period": 0,
      "async_checkpointing": False,
      "dataset_type": "tfds",
      "lora": None,
      "checkpoint_storage_target_data_file_size_bytes": checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
      "elastic_enabled": False,
      "load_parameters_path": "",
      "load_full_state_path": "",
      "checkpoint_storage_concurrent_gb": 8,
      "enable_single_replica_ckpt_restoring": False,
      "checkpoint_storage_use_ocdbt": True,
      "checkpoint_storage_use_zarr3": True,
      "enable_orbax_v1": False,
      "checkpoint_conversion_fn": None,
      "source_checkpoint_layout": "orbax",
      "expansion_factor_real_data": -1,
      "scan_layers": False,
  }
  base.update(overrides)
  return SimpleNamespace(**base)


class TestPickDeviceResidentOptimizerLeaves(unittest.TestCase):
  """Covers greedy largest-first packing, backfill, scalar exclusion, tie-breaking, and Muon MaskedNode."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def test_zero_partial_backfill_overflow_and_scalar_exclusion(self):
    devices = np.array(jax.devices()[:2])
    mesh = Mesh(devices, ("fsdp",))
    sharded = NamedSharding(mesh, P("fsdp", None))
    replicated = NamedSharding(mesh, P())

    # Per-device sizes on 2-device fsdp mesh:
    # - big_mu:  (2048, 256) float32 sharded on dim 0 -> (1024, 256) * 4 = 1,048,576 B (1.0 MiB)
    # - mid_nu:  (1024, 256) float32 sharded on dim 0 -> (512, 256)  * 4 =   524,288 B (0.5 MiB)
    # - bias_mu: (256,) float32 replicated            -> 256         * 4 =     1,024 B (1.0 KiB)
    # - step & count: () scalars                      -> 0 B (always excluded from resident set)
    opt_abs = {
        "step": jax.ShapeDtypeStruct((), jnp.uint32, sharding=replicated),
        "opt_state": {
            0: {
                "count": jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated),
                "mu": {
                    "big": jax.ShapeDtypeStruct((2048, 256), jnp.float32, sharding=sharded),
                    "bias": jax.ShapeDtypeStruct((256,), jnp.float32, sharding=replicated),
                },
                "nu": {
                    "mid": jax.ShapeDtypeStruct((1024, 256), jnp.float32, sharding=sharded),
                },
            }
        },
    }
    opt_shard = {
        "step": replicated,
        "opt_state": {
            0: {
                "count": replicated,
                "mu": {"big": sharded, "bias": replicated},
                "nu": {"mid": sharded},
            }
        },
    }

    # 1. Budget = 0.0 -> 0 resident leaves, all non-scalar bytes offloaded.
    res0, b_res0, b_off0 = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs, opt_shard, 0.0)
    self.assertEqual(res0, set())
    self.assertEqual(b_res0, 0)
    self.assertEqual(b_off0, 1_048_576 + 524_288 + 1_024)

    # 2. Partial budget = 1.25 MiB -> selects 1.0 MiB ("big"), skips 0.5 MiB ("mid" would hit 1.5 MiB),
    #    and backfills 1.0 KiB ("bias", 1.0 MiB + 1.0 KiB <= 1.25 MiB). Scalars stay excluded.
    budget_gib = (1_048_576 + 262_144) / float(1 << 30)
    res_part, b_res_part, b_off_part = train_state_nnx.pick_device_resident_optimizer_leaves(
        opt_abs, opt_shard, budget_gib
    )
    self.assertEqual(
        res_part,
        {
            ("opt_state", 0, "mu", "big"),
            ("opt_state", 0, "mu", "bias"),
        },
    )
    self.assertNotIn(("step",), res_part)
    self.assertNotIn(("opt_state", 0, "count"), res_part)
    self.assertEqual(b_res_part, 1_048_576 + 1_024)
    self.assertEqual(b_off_part, 524_288)

    # 3. Overflow budget (2.0 GiB) -> keeps all 3 non-scalar leaves on device, scalars still excluded.
    res_all, b_res_all, b_off_all = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs, opt_shard, 2.0)
    self.assertEqual(
        res_all,
        {
            ("opt_state", 0, "mu", "big"),
            ("opt_state", 0, "nu", "mid"),
            ("opt_state", 0, "mu", "bias"),
        },
    )
    self.assertNotIn(("step",), res_all)
    self.assertNotIn(("opt_state", 0, "count"), res_all)
    self.assertEqual(b_res_all, 1_048_576 + 524_288 + 1_024)
    self.assertEqual(b_off_all, 0)

  def test_deterministic_canonical_path_tie_breaking_and_invalid_budgets(self):
    mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))
    replicated = NamedSharding(mesh, P())

    opt_abs_a = {
        "a": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "b": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "c": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
    }
    opt_abs_b = {
        "c": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "b": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "a": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
    }
    opt_shard_a = {"a": replicated, "b": replicated, "c": replicated}
    # Also omit "b" from opt_shard_b (None) to verify key-based sharding lookup never misaligns leaves:
    opt_shard_b = {"c": replicated, "b": None, "a": replicated}

    two_leaf_gib = (2 * 256 * 256 * 4) / float(1 << 30)
    pick_a, b_a, off_a = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs_a, opt_shard_a, two_leaf_gib)
    pick_b, b_b, off_b = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs_b, opt_shard_b, two_leaf_gib)
    self.assertEqual(pick_a, {("a",), ("b",)})
    self.assertEqual(pick_b, {("a",), ("b",)})
    self.assertEqual(b_a, b_b)
    self.assertEqual(off_a, off_b)

    for bad_gib in (-0.1, float("nan"), float("inf"), float("-inf")):
      with self.subTest(bad_gib=bad_gib):
        with self.assertRaises(ValueError):
          train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs_a, opt_shard_a, bad_gib)

  def test_device_resident_optimizer_mask_builds_boolean_tree_and_ordered_chosen(self):
    class HalfShard:
      """Duck-typed sharding that halves the leading dimension per device."""

      def shard_shape(self, shape):
        return (shape[0] // 2,) + tuple(shape[1:])

    class WholeShard:
      """Duck-typed replicated sharding."""

    opt_tree = {
        "a": jax.ShapeDtypeStruct((8,), jnp.float32),  # 32 B whole, 16 B under HalfShard
        "b": jax.ShapeDtypeStruct((4,), jnp.float32),  # 16 B whole
        "c": jax.ShapeDtypeStruct((6,), jnp.float32),  # 24 B whole
        "step": jax.ShapeDtypeStruct((), jnp.uint32),  # scalar -> never resident
    }
    whole_shards = {k: WholeShard() for k in opt_tree}
    mask, chosen, res_b, off_b = train_state_nnx.device_resident_optimizer_mask(
        opt_tree, whole_shards, 52 / float(1 << 30)
    )
    # 32 B ('a') fits; 24 B ('c') overflows 52 B; 16 B ('b') backfills (32 + 16 = 48 <= 52):
    self.assertEqual(mask, {"a": True, "b": True, "c": False, "step": False})
    self.assertEqual(chosen, [("['a']", 32), ("['b']", 16)])
    self.assertEqual(res_b, 48)
    self.assertEqual(off_b, 24)

    # With HalfShard on 'a' (16 B), 'c' (24 B) is largest and 'a' (16 B) wins tie over 'b' (16 B):
    half_a_shards = {**whole_shards, "a": HalfShard()}
    mask2, chosen2, res_b2, off_b2 = train_state_nnx.device_resident_optimizer_mask(
        opt_tree, half_a_shards, 40 / float(1 << 30)
    )
    self.assertEqual(mask2, {"a": True, "b": False, "c": True, "step": False})
    self.assertEqual(chosen2, [("['c']", 24), ("['a']", 16)])
    self.assertEqual(res_b2, 40)
    self.assertEqual(off_b2, 16)

  def test_muon_masked_node_optimizer_partitioning(self):
    mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))
    cfg = _state_config(
        optimizer_memory_host_offload=True,
        optimizer_device_resident_gib=(32 * 16 * 4 // 2 + 512) / float(1 << 30),
    )

    def init_fn():
      model = _TinyBudgetModel(nnx.Rngs(0))
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax_muon(1e-3), wrt=nnx.Param))

    _, _, shardings = maxtext_utils.get_abstract_state_nnx(cfg, mesh, init_fn, is_training=True)
    opt_kinds = {
        s.memory_kind for s in jax.tree.leaves(shardings.optimizer, is_leaf=lambda x: isinstance(x, NamedSharding))
    }
    self.assertEqual(opt_kinds, {"device", "pinned_host"})


class TestPartitionOptimizerMemoryShardingsAndZeRO1(unittest.TestCase):
  """Covers explicit memory_kind='device' tagging on non-offloaded leaves and ZeRO-1 interaction."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def test_partition_tags_non_offloaded_leaves_with_device_and_splits_optimizer(self):
    for mesh in (
        jax.sharding.AbstractMesh((2,), ("fsdp",)),
        Mesh(np.array(jax.devices()[:2]), ("fsdp",)),
    ):
      with self.subTest(mesh_type=type(mesh).__name__):
        untagged_sharded = NamedSharding(mesh, P("fsdp", None))
        untagged_repl = NamedSharding(mesh, P())
        if isinstance(mesh, jax.sharding.AbstractMesh):
          self.assertIsNone(untagged_sharded.memory_kind)
          self.assertIsNone(untagged_repl.memory_kind)

        model = _TinyBudgetModel(nnx.Rngs(5))
        ts = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param))
        _, state = nnx.split(ts)

        def _to_sds(arr, sharded_s=untagged_sharded, repl_s=untagged_repl):
          s = sharded_s if arr.ndim == 2 and arr.shape[0] % 2 == 0 else repl_s
          return jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=s)

        abstract_state = jax.tree.map(_to_sds, state)
        state_mesh_shardings = jax.tree.map(
            lambda x: x.sharding, abstract_state, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)
        )

        # Budget 1,500 B/device keeps ONE 1,024 B leaf + 64 B bias leaves on device
        # and offloads remaining 1,024 B & 512 B leaves + scalar step/count to pinned_host:
        budget_gib = 1500 / float(1 << 30)
        partitioned = train_state_nnx.partition_optimizer_memory_shardings(
            abstract_state, state_mesh_shardings, device_resident_gib=budget_gib
        )

        # Non-offloaded model leaves must be explicitly tagged memory_kind="device":
        model_kinds = {
            s.memory_kind for s in jax.tree.leaves(partitioned.model, is_leaf=lambda x: isinstance(x, NamedSharding))
        }
        self.assertEqual(model_kinds, {"device"})

        # Optimizer leaves must mix "device" (resident non-scalar) and "pinned_host" (offloaded + scalar step/count):
        opt_kinds = {
            s.memory_kind for s in jax.tree.leaves(partitioned.optimizer, is_leaf=lambda x: isinstance(x, NamedSharding))
        }
        self.assertEqual(opt_kinds, {"device", "pinned_host"})
        self.assertEqual(partitioned.optimizer["step"].get_value().memory_kind, "pinned_host")

  def test_zero1_sharding_preserves_model_memory_kind_device_and_pinned_host(self):
    devices = np.array(jax.devices()[:4]).reshape(2, 2)
    mesh = Mesh(devices, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    def init_fn():
      model = _TinyBudgetModel(nnx.Rngs(0))
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param))

    # 1. parameter_memory_host_offload=False -> model memory_kind stays "device" across all params
    #    even when optimizer leaves mix "device" and "pinned_host" under ZeRO-1:
    cfg_dev = _state_config(
        shard_optimizer_over_data=True,
        optimizer_memory_host_offload=True,
        optimizer_device_resident_gib=700 / float(1 << 30),  # keeps 512 B/dev embed mu, offloads nu & others
        parameter_memory_host_offload=False,
    )
    _, _, shardings_dev = maxtext_utils.get_abstract_state_nnx(cfg_dev, mesh, init_fn, is_training=True)
    opt_kinds_dev = {
        s.memory_kind for s in jax.tree.leaves(shardings_dev.optimizer, is_leaf=lambda x: isinstance(x, NamedSharding))
    }
    self.assertEqual(opt_kinds_dev, {"device", "pinned_host"})

    _, updated_dev = sharding.maybe_update_params_sharding_with_opt(cfg_dev, shardings_dev)
    model_kinds_dev = {
        s.memory_kind for s in jax.tree.leaves(updated_dev.model, is_leaf=lambda x: isinstance(x, NamedSharding))
    }
    self.assertEqual(model_kinds_dev, {"device"})
    # Verify ZeRO-1 actually added "data" to embed's spec:
    embed_spec = updated_dev.model["embed"].get_value().spec
    self.assertIn("data", jax.tree.leaves(tuple(embed_spec)))

    # 2. parameter_memory_host_offload=True -> model Param memory_kind stays "pinned_host"
    #    even for params whose optimizer `mu` leaf is kept on "device":
    cfg_param_off = _state_config(
        shard_optimizer_over_data=True,
        optimizer_memory_host_offload=True,
        optimizer_device_resident_gib=700 / float(1 << 30),
        parameter_memory_host_offload=True,
        param_scan_axis=0,
    )
    _, _, shardings_param_off = maxtext_utils.get_abstract_state_nnx(cfg_param_off, mesh, init_fn, is_training=True)
    _, updated_param_off = sharding.maybe_update_params_sharding_with_opt(cfg_param_off, shardings_param_off)
    _, param_only, _ = nnx.split(updated_param_off.model, nnx.Param, ...)
    param_kinds = {s.memory_kind for s in jax.tree.leaves(param_only, is_leaf=lambda x: isinstance(x, NamedSharding))}
    self.assertEqual(param_kinds, {"pinned_host"})

    # 3. At default optimizer_device_resident_gib=0.0, ZeRO-1 replaces model shardings
    #    with `mu` shardings identically to main (avoiding host/device GA mismatches):
    cfg_param_only_off = _state_config(
        shard_optimizer_over_data=True,
        optimizer_memory_host_offload=False,
        optimizer_device_resident_gib=0.0,
        parameter_memory_host_offload=True,
        param_scan_axis=0,
    )
    _, _, shardings_param_only_off = maxtext_utils.get_abstract_state_nnx(
        cfg_param_only_off, mesh, init_fn, is_training=True
    )
    _, updated_param_only_off = sharding.maybe_update_params_sharding_with_opt(
        cfg_param_only_off, shardings_param_only_off
    )
    _, param_only_default, _ = nnx.split(updated_param_only_off.model, nnx.Param, ...)
    param_only_kinds = {
        s.memory_kind for s in jax.tree.leaves(param_only_default, is_leaf=lambda x: isinstance(x, NamedSharding))
    }
    self.assertEqual(param_only_kinds, {"device"})


class TestOrbaxCheckpointRoundTripWithOptBudget(unittest.TestCase):
  """End-to-end Orbax checkpoint save & restore with optimizer_memory_host_offload=True and partial budget."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)
    self._tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
    self._mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))

  def _make_manager(self, subdir: str):
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._tmpdir, subdir),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )

  def _init_fn(self, seed: int = 0):
    tx = optax.adamw(learning_rate=1e-3, weight_decay=0.01)

    def fn():
      model = _TinyBudgetModel(nnx.Rngs(seed))
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    return fn

  def test_cast_step_on_pinned_host_round_trip(self):
    host_sharding = NamedSharding(self._mesh, P(), memory_kind="pinned_host")
    step_u32 = jax.device_put(jnp.array(7, dtype=jnp.uint32), host_sharding)
    self.assertEqual(step_u32.sharding.memory_kind, "pinned_host")

    casted_i32 = train_state_nnx._cast_step(step_u32, jnp.int32)  # pylint: disable=protected-access
    self.assertEqual(casted_i32.dtype, jnp.int32)
    self.assertEqual(int(casted_i32), 7)

    step_i32_host = jax.device_put(casted_i32, host_sharding)
    casted_u32 = train_state_nnx._cast_step(step_i32_host, jnp.uint32)  # pylint: disable=protected-access
    self.assertEqual(casted_u32.dtype, jnp.uint32)
    self.assertEqual(casted_u32.sharding.memory_kind, "pinned_host")
    self.assertEqual(int(casted_u32), 7)

  def test_orbax_save_and_restore_with_partial_device_resident_budget(self):
    # Partial budget keeps largest leaf(s) on device and offloads the rest + scalar `step` to pinned_host:
    cfg = _state_config(
        optimizer_memory_host_offload=True,
        optimizer_device_resident_gib=1500 / float(1 << 30),
    )
    mgr = self._make_manager("opt_budget_ckpt")
    init_fn = self._init_fn(seed=11)

    with mock.patch.object(maxtext_utils.lineage_adapter, "is_native", return_value=False):
      state, _, state_mesh_shardings, *_ = maxtext_utils.setup_initial_state(
          None, cfg, self._mesh, mgr, init_fn, is_training=True
      )

    # Verify concrete initialized state has mixed {"device", "pinned_host"} optimizer leaves and pinned_host step:
    init_opt_kinds = {
        arr.sharding.memory_kind for arr in jax.tree.leaves(state["optimizer"]) if isinstance(arr, jax.Array)
    }
    self.assertEqual(init_opt_kinds, {"device", "pinned_host"})
    self.assertEqual(state["optimizer"]["step"].get_value().sharding.memory_kind, "pinned_host")

    graphdef = nnx.graphdef(init_fn())
    device_opt_shardings = jax.tree.map(
        lambda s: s.with_memory_kind("device") if isinstance(s, NamedSharding) else s,
        state_mesh_shardings.optimizer,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )

    @jax.jit(in_shardings=(state_mesh_shardings, None), out_shardings=state_mesh_shardings, donate_argnums=(0,))
    def _step(st, tokens):
      ts = nnx.merge(graphdef, st)
      opt_st = nnx.state(ts.optimizer)
      moved_opt = jax.tree.map(
          lambda arr, src_s, dst_s: (
              jax.device_put(arr, dst_s)
              if isinstance(src_s, NamedSharding) and src_s.memory_kind == "pinned_host"
              else arr
          ),
          opt_st,
          state_mesh_shardings.optimizer,
          device_opt_shardings,
      )
      nnx.update(ts.optimizer, moved_opt)
      grads = nnx.grad(lambda m: jnp.mean(m(tokens) ** 2))(ts.model)
      ts.apply_gradients(grads)
      return nnx.state(ts)

    tokens = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    with jax.set_mesh(self._mesh):
      for _ in range(3):
        state = _step(state, tokens)

    self.assertEqual(int(state["optimizer"]["step"].get_value()), 3)
    self.assertEqual(state["optimizer"]["step"].get_value().sharding.memory_kind, "pinned_host")

    ts_to_save = nnx.merge(graphdef, state)
    checkpointing.save_checkpoint(mgr, 3, ts_to_save, config=cfg, force=True)
    checkpointing.wait_until_finished(mgr)

    # Verify raw on-disk Orbax checkpoint is standard Linen layout with step == 3 (int32):
    raw_loaded = ocp.load(os.path.join(self._tmpdir, "opt_budget_ckpt", "3"), checkpointable_name="items")
    self.assertEqual(int(raw_loaded["step"]), 3)

    # Advance reference state to step 4:
    next_tokens = jnp.array([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=jnp.int32)
    with jax.set_mesh(self._mesh):
      expected_step4 = _step(
          jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, state),
          next_tokens,
      )

    # Restore into fresh model (different seed) under the same partial budget and verify bit-exact match + step 4:
    init_fresh = self._init_fn(seed=999)
    with mock.patch.object(maxtext_utils.lineage_adapter, "is_native", return_value=False):
      restored_state, *_ = maxtext_utils.setup_initial_state(None, cfg, self._mesh, mgr, init_fresh, is_training=True)

    restored_opt_kinds = {
        arr.sharding.memory_kind for arr in jax.tree.leaves(restored_state["optimizer"]) if isinstance(arr, jax.Array)
    }
    self.assertEqual(restored_opt_kinds, {"device", "pinned_host"})
    self.assertEqual(int(restored_state["optimizer"]["step"].get_value()), 3)
    for exp_leaf, got_leaf in zip(jax.tree.leaves(state), jax.tree.leaves(restored_state)):
      np.testing.assert_array_equal(np.asarray(got_leaf), np.asarray(exp_leaf))

    with jax.set_mesh(self._mesh):
      actual_step4 = _step(restored_state, next_tokens)
    for exp_leaf, got_leaf in zip(jax.tree.leaves(expected_step4), jax.tree.leaves(actual_step4)):
      np.testing.assert_array_equal(np.asarray(got_leaf), np.asarray(exp_leaf))


class TestOptimizerDeviceResidentConfigAndFailFast(unittest.TestCase):
  """Covers pyconfig, RLConfig, cross-field trainer exclusions, and fail-fast entrypoints."""

  def test_pyconfig_rlconfig_and_cross_field_validation(self):
    cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "run_name=test_opt_budget",
            "optimizer_memory_host_offload=True",
            "optimizer_device_resident_gib=0.25",
        ]
    )
    self.assertTrue(cfg.optimizer_memory_host_offload)
    self.assertAlmostEqual(cfg.optimizer_device_resident_gib, 0.25)

    # Rejects > 0 when optimizer_memory_host_offload=False:
    with self.assertRaisesRegex(ValidationError, "requires optimizer_memory_host_offload=True"):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_no_offload",
              "optimizer_memory_host_offload=False",
              "optimizer_device_resident_gib=1.0",
          ]
      )

    # Rejects negative, NaN, +Inf, -Inf on Optimizer validator:
    for bad_val in ("-0.5", "nan", "inf", "-inf"):
      with self.subTest(bad_val=bad_val):
        with self.assertRaisesRegex(ValidationError, "optimizer_device_resident_gib must be a finite number >= 0"):
          pyconfig.initialize(
              [
                  "",
                  get_test_config_path(),
                  "run_name=test_bad_num",
                  "optimizer_memory_host_offload=True",
                  f"optimizer_device_resident_gib={bad_val}",
              ]
          )

    # Rejects unsupported trainers on MaxTextConfig:
    for extra_args, match_pat in (
        (["lora.enable_lora=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"], "LoRA"),
        (
            ["enable_diloco=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"],
            "DiLoCo",
        ),
        (["use_sft=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"], "use_sft"),
        (
            [
                "use_dpo=True",
                "packing=False",
                "optimizer_memory_host_offload=True",
                "optimizer_device_resident_gib=0.25",
            ],
            "use_dpo",
        ),
        (["use_grpo=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"], "use_grpo"),
    ):
      with self.subTest(extra_args=extra_args):
        with self.assertRaisesRegex(ValidationError, match_pat):
          pyconfig.initialize(["", get_test_config_path(), "run_name=test_unsupported"] + extra_args)

    # RLConfig inherits optimizer_device_resident_gib and rejects > 0 while pointing to MaxTextTrainingEngine:
    self.assertIn("optimizer_device_resident_gib", config_types.RLConfig.model_fields)
    rl_yml_path = os.path.join(os.path.dirname(get_test_config_path()), "post_train", "rl.yml")
    with self.assertRaisesRegex(ValidationError, "MaxTextTrainingEngine"):
      pyconfig.initialize_pydantic(
          ["", rl_yml_path, "model_name=qwen3-0.6b", "optimizer_device_resident_gib=0.25"],
          config_class=config_types.RLConfig,
      )

  def test_fail_fast_entrypoints(self):
    # 1. lora_utils.setup_initial_lora_state:
    budget_ns = SimpleNamespace(optimizer_device_resident_gib=0.5)
    with self.assertRaisesRegex(ValueError, "LoRA does not support optimizer_device_resident_gib"):
      lora_utils.setup_initial_lora_state(None, optax.sgd(1e-3), budget_ns, None, None, "/tmp/fake_lora")

    # 2. train_distill rejects optimizer_device_resident_gib > 0 on student or teacher:
    if importlib.util.find_spec("tunix") is not None:
      from maxtext.trainers.post_train.distillation import train_distill  # pylint: disable=import-outside-toplevel

      ok_ns = SimpleNamespace(optimizer_device_resident_gib=0.0, vocab_size=32)
      bad_ns = SimpleNamespace(optimizer_device_resident_gib=0.5, vocab_size=32)
      with self.assertRaisesRegex(
          ValueError, "Distillation \\(student_config\\) does not support optimizer_device_resident_gib"
      ):
        train_distill.train_distill(bad_ns, ok_ns)
      with self.assertRaisesRegex(
          ValueError, "Distillation \\(teacher_config\\) does not support optimizer_device_resident_gib"
      ):
        train_distill.train_distill(ok_ns, bad_ns)

    # 3. MaxTextTrainingEngine supports optimizer_device_resident_gib > 0 with optimizer_memory_host_offload=True:
    if importlib.util.find_spec("tunix") is not None and importlib.util.find_spec("tunix.dist_rl") is not None:
      from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

      with tempfile.TemporaryDirectory() as tmpdir:
        cfg_eng = pyconfig.initialize(
            [None, get_test_config_path()],
            model_name="llama3.1-8b",
            run_name="test_eng_budget",
            base_output_directory=tmpdir,
            tensorboard_dir=tmpdir,
            enable_tensorboard=False,
            enable_checkpointing=False,
            skip_jax_distributed_system=True,
            optimizer_memory_host_offload=True,
            optimizer_device_resident_gib=1500 / float(1 << 30),
        )
        mesh = Mesh(maxtext_utils.create_device_mesh(cfg_eng), cfg_eng.mesh_axes)
        with (
            mock.patch.object(
                maxtext_engine.model_creation_utils,
                "from_pretrained",
                return_value=(_TinyBudgetModel(nnx.Rngs(0)), mesh),
            ),
            mock.patch.object(
                maxtext_engine.train_utils,
                "create_training_optimizer",
                return_value=(lambda step: jnp.array(1e-3), optax.adamw(1e-3)),
            ),
        ):
          eng = maxtext_engine.MaxTextTrainingEngine(cfg_eng)
          opt_kinds = {
              leaf.sharding.memory_kind
              for leaf in jax.tree.leaves(nnx.state(eng.state.optimizer))
              if isinstance(leaf, jax.Array)
          }
          self.assertEqual(opt_kinds, {"device", "pinned_host"})

          cfg_no_offload = pyconfig.initialize(
              [None, get_test_config_path()],
              model_name="llama3.1-8b",
              run_name="test_eng_no_offload",
              base_output_directory=tmpdir,
              tensorboard_dir=tmpdir,
              enable_tensorboard=False,
              enable_checkpointing=False,
              skip_jax_distributed_system=True,
              optimizer_memory_host_offload=False,
          )
          cfg_no_offload.get_keys()["optimizer_device_resident_gib"] = 0.25
          with self.assertRaisesRegex(ValueError, "requires optimizer_memory_host_offload=True"):
            maxtext_engine.MaxTextTrainingEngine(cfg_no_offload)


if __name__ == "__main__":
  unittest.main()

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

"""Unit tests for `MaxTextTrainingEngine` mesh sharing and per-leaf `NamedSharding` preservation (`M4E-10`)."""

import contextlib
import sys
import types
from typing import Any
import unittest

from absl.testing import absltest
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import nnx_wrappers  # pylint: disable=unused-import
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.utils import model_creation_utils
import numpy as np
from tests.utils.test_helpers import get_test_config_path

_TE_PKG_NAME = f"{pyconfig.__name__.rsplit('.configs.', 1)[0]}.training_engine"

try:
  from maxtext.training_engine import maxtext_engine

  HAS_TUNIX = True
except ImportError:
  maxtext_engine = None
  HAS_TUNIX = False


@contextlib.contextmanager
def _with_tunix_stub_if_needed():
  """Provides minimal Tunix dataclass stubs when running in `cpu-unit` without `tunix` installed."""
  if HAS_TUNIX:
    yield maxtext_engine
    return

  tunix_mod = types.ModuleType("tunix")
  dist_rl = types.ModuleType("tunix.dist_rl")
  common = types.ModuleType("tunix.dist_rl.common")
  datatypes = types.ModuleType("tunix.dist_rl.common.datatypes")
  sft = types.ModuleType("tunix.sft")
  sft_utils = types.ModuleType("tunix.sft.utils")

  class _TrainerPayload:
    pass

  @flax.struct.dataclass
  class _RLTrainerPayload(_TrainerPayload):
    pass

  @flax.struct.dataclass
  class _LossOutput:
    primary_loss: object = None
    aux_metrics: object = None

  @flax.struct.dataclass
  class _WeightedMetric:
    """Minimal WeightedMetric stub for torch-free MaxTextTrainingEngine tests."""

    unreduced_sum: object
    denominator: object
    eps: float = flax.struct.field(pytree_node=False, default=1e-8)
    min_denom: float = flax.struct.field(pytree_node=False, default=0.0)

    @classmethod
    def from_mean_and_weight(cls, mean, weight):
      return cls(unreduced_sum=mean * weight, denominator=weight)

    def compute(self):
      return self.unreduced_sum / self.denominator

    def compute_scale(self):
      return self.denominator

  datatypes.TrainerPayload = _TrainerPayload
  datatypes.RLTrainerPayload = _RLTrainerPayload
  sft_utils.LossOutput = _LossOutput
  sft_utils.WeightedMetric = _WeightedMetric
  stub_modules = {
      "tunix": tunix_mod,
      "tunix.dist_rl": dist_rl,
      "tunix.dist_rl.common": common,
      "tunix.dist_rl.common.datatypes": datatypes,
      "tunix.sft": sft,
      "tunix.sft.utils": sft_utils,
  }
  engine_submodules = ("abstract_engine", "maxtext_engine", "metrics")
  tracked_keys = (*stub_modules, *(f"{_TE_PKG_NAME}.{attr}" for attr in engine_submodules))
  saved_sys = {key: sys.modules[key] for key in tracked_keys if key in sys.modules}
  pkg = sys.modules.get(_TE_PKG_NAME)
  saved_pkg_attrs = {attr: getattr(pkg, attr) for attr in engine_submodules if pkg is not None and hasattr(pkg, attr)}
  try:
    sys.modules.update(stub_modules)
    for attr in engine_submodules:
      sys.modules.pop(f"{_TE_PKG_NAME}.{attr}", None)
      if pkg is not None and hasattr(pkg, attr):
        delattr(pkg, attr)
    from maxtext.training_engine import maxtext_engine as stubbed_engine  # pylint: disable=import-outside-toplevel

    yield stubbed_engine
  finally:
    for key in tracked_keys:
      if key in saved_sys:
        sys.modules[key] = saved_sys[key]
      else:
        sys.modules.pop(key, None)
    pkg_after = sys.modules.get(_TE_PKG_NAME)
    if pkg_after is not None:
      for attr in engine_submodules:
        if attr in saved_pkg_attrs:
          setattr(pkg_after, attr, saved_pkg_attrs[attr])
        elif hasattr(pkg_after, attr):
          delattr(pkg_after, attr)


def _make_engine_cfg(output_dir: str, **overrides: Any):
  """Builds a small 4-CPU m3 Qwen3 config for engine sharding tests."""
  argv = [
      "",
      get_test_config_path("base.yml"),
      f"base_output_directory={output_dir}",
      "run_name=m3_engine_sharding_test",
      "model_name=qwen3-0.6b",
      "use_m3_model=true",
      "override_model_config=true",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=16",
      "vocab_size=256",
      "ici_fsdp_parallelism=2",
      "ici_tensor_parallelism=2",
      "per_device_batch_size=1.0",
      "max_target_length=8",
      "dtype=float32",
      "weight_dtype=float32",
      "enable_checkpointing=false",
      "use_weight_converter=false",
      "skip_jax_distributed_system=true",
  ]
  for key, value in overrides.items():
    argv.append(f"{key}={value}")
  return pyconfig.initialize(argv)


def _assert_engine_preserves_m3_sharding(tc: absltest.TestCase, engine: Any, mesh: jax.sharding.Mesh) -> None:
  """Verifies `engine._mesh == mesh` and multi-D parameter leaves keep non-replicated `NamedSharding`."""
  tc.assertEqual(engine._mesh, mesh)  # pylint: disable=protected-access
  flat_params = nnx.to_flat_state(nnx.state(engine.model, nnx.Param))
  tc.assertNotEmpty(flat_params)

  non_replicated_multi_d = 0
  for path, var in flat_params:
    arr = var.get_raw_value()
    leaf_sharding = arr.sharding
    engine_sharding = engine._mesh_sharding(arr)  # pylint: disable=protected-access
    tc.assertIsInstance(leaf_sharding, jax.sharding.NamedSharding, f"Non-NamedSharding at {path}")
    tc.assertEqual(leaf_sharding.mesh, mesh, f"Mesh mismatch at {path}")
    tc.assertEqual(engine_sharding, leaf_sharding, f"_mesh_sharding mismatch at {path}")
    if arr.ndim >= 2 and "scale" not in path:
      tc.assertNotEqual(
          leaf_sharding.spec,
          jax.sharding.PartitionSpec(),
          f"Multi-D weight {path} silently fell back to replicated PartitionSpec()",
      )
      tc.assertTrue(
          any(axis is not None for axis in leaf_sharding.spec),
          f"Multi-D weight {path} has all-None PartitionSpec {leaf_sharding.spec}",
      )
      non_replicated_multi_d += 1
  tc.assertGreater(non_replicated_multi_d, 0)


class EngineShardingTest(absltest.TestCase):
  """Verifies `MaxTextTrainingEngine` mesh identity and per-leaf `NamedSharding` on `m3` (`M4E-10`)."""

  @unittest.skipUnless(HAS_TUNIX, "tunix not installed")
  def test_engine_preserves_mesh_and_named_sharding_with_native_tunix(self):
    # Importing `maxtext.layers.nnx_wrappers` at module load disables global Flax eager sharding.
    self.assertFalse(nnx.using_eager_sharding())
    out_dir = self.create_tempdir().full_path
    cfg = _make_engine_cfg(out_dir)
    mesh = mesh_from_config(cfg)
    engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=False)
    _assert_engine_preserves_m3_sharding(self, engine, mesh)

  def test_engine_and_from_pretrained_preserve_sharding_after_nnx_wrappers_import(self):
    # Ensure global Flax eager sharding is disabled inside a scoped context without leaking.
    prev_flax_flag = flax.config.flax_always_shard_variable
    prev_using = nnx.using_eager_sharding()
    with nnx.use_eager_sharding(False):
      self.assertFalse(nnx.using_eager_sharding())
      out_dir = self.create_tempdir().full_path
      cfg = _make_engine_cfg(out_dir)
      mesh = mesh_from_config(cfg)

      model = model_creation_utils.from_pretrained(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          rng_key=jax.random.PRNGKey(0),
          wrap_with_tunix_adapter=False,
      )
      wi0_sharding = model.decoder.layers.mlp.wi_0.kernel.get_raw_value().sharding
      self.assertIsInstance(wi0_sharding, jax.sharding.NamedSharding)
      self.assertEqual(wi0_sharding.mesh, mesh)
      self.assertNotEqual(wi0_sharding.spec, jax.sharding.PartitionSpec())

      with _with_tunix_stub_if_needed() as eng_mod:
        engine = eng_mod.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=False)
        _assert_engine_preserves_m3_sharding(self, engine, mesh)
    self.assertEqual(flax.config.flax_always_shard_variable, prev_flax_flag)
    self.assertEqual(nnx.using_eager_sharding(), prev_using)

  def test_negative_control_missing_eager_sharding_or_mismatched_mesh_falls_back_to_replicated(self):
    prev_flax_flag = flax.config.flax_always_shard_variable
    prev_using = nnx.using_eager_sharding()
    with nnx.use_eager_sharding(False):
      self.assertFalse(nnx.using_eager_sharding())
      out_dir = self.create_tempdir().full_path
      cfg = _make_engine_cfg(out_dir)
      mesh = mesh_from_config(cfg)

      # Negative control 1: without `eager_sharding=True`, `nnx.with_partitioning` leaves the array
      # replicated (`PartitionSpec()`) when `nnx_wrappers` has disabled global eager sharding.
      with jax.set_mesh(mesh):
        unsharded_init = nnx.with_partitioning(
            nnx.initializers.normal(),
            jax.sharding.PartitionSpec("fsdp", "tensor"),
        )
        lazy_param = nnx.Param(unsharded_init(jax.random.PRNGKey(0), (64, 128), jnp.float32))
      self.assertEqual(lazy_param.get_raw_value().sharding.spec, jax.sharding.PartitionSpec())

      # Negative control 2: if a parameter leaf is on a different mesh object than `engine._mesh`,
      # `engine._mesh_sharding(leaf)` falls back to replicated `PartitionSpec()` and fails the check.
      other_mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]).reshape((2, 1)), ("fsdp", "tensor"))
      other_sharding = jax.sharding.NamedSharding(other_mesh, jax.sharding.PartitionSpec("fsdp", "tensor"))
      mismatched_leaf = jax.device_put(jnp.ones((64, 128), dtype=jnp.float32), other_sharding)
      with _with_tunix_stub_if_needed() as eng_mod:
        engine = eng_mod.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=False)
        resolved = engine._mesh_sharding(mismatched_leaf)  # pylint: disable=protected-access
        self.assertEqual(resolved.spec, jax.sharding.PartitionSpec())
        engine.model.decoder.layers.mlp.wi_0.kernel.set_raw_value(mismatched_leaf)
        with self.assertRaises(AssertionError):
          _assert_engine_preserves_m3_sharding(self, engine, mesh)
    self.assertEqual(flax.config.flax_always_shard_variable, prev_flax_flag)
    self.assertEqual(nnx.using_eager_sharding(), prev_using)

  def test_eager_sharding_and_tunix_stub_restore_global_state(self):
    pkg = sys.modules[_TE_PKG_NAME]
    had_eng_before = hasattr(pkg, "maxtext_engine")
    had_abs_before = hasattr(pkg, "abstract_engine")
    had_metrics_before = hasattr(pkg, "metrics")
    orig_flax_flag = flax.config.flax_always_shard_variable
    orig_stack = list(nnx.use_eager_sharding.get_stack())
    orig_using = nnx.using_eager_sharding()
    try:
      flax.config.update("flax_always_shard_variable", True)
      with nnx.use_eager_sharding(True):
        self.assertTrue(nnx.using_eager_sharding())
        self.assertTrue(flax.config.flax_always_shard_variable)
        self.test_engine_and_from_pretrained_preserve_sharding_after_nnx_wrappers_import()
        self.assertTrue(nnx.using_eager_sharding())
        self.assertTrue(flax.config.flax_always_shard_variable)
        self.test_negative_control_missing_eager_sharding_or_mismatched_mesh_falls_back_to_replicated()
        self.assertTrue(nnx.using_eager_sharding())
        self.assertTrue(flax.config.flax_always_shard_variable)
    finally:
      flax.config.update("flax_always_shard_variable", orig_flax_flag)
    self.assertEqual(flax.config.flax_always_shard_variable, orig_flax_flag)
    self.assertEqual(list(nnx.use_eager_sharding.get_stack()), orig_stack)
    self.assertEqual(nnx.using_eager_sharding(), orig_using)
    self.assertEqual(hasattr(pkg, "maxtext_engine"), had_eng_before)
    self.assertEqual(hasattr(pkg, "abstract_engine"), had_abs_before)
    self.assertEqual(hasattr(pkg, "metrics"), had_metrics_before)


if __name__ == "__main__":
  absltest.main()

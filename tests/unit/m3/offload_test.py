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

"""Unit tests for `maxtext.m3.infra.offload` and runtime TPU HBM telemetry."""

import contextlib
import io
import os
import re
from unittest import mock
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.topologies import get_topology_desc
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3.core.sharding import TensorType, mesh_from_config, sharding_from_config
from maxtext.m3.infra import offload as m3_offload
from maxtext.m3.infra import quantization as m3_quant
from maxtext.m3.train import loop
import optax
import pytest

from maxtext.trainers.pre_train import train as legacy_train
from maxtext.trainers.pre_train import train_compile
from maxtext.training_engine import memory_report as te_memory_report
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import memory_report
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding as sharding_utils
from maxtext.utils import train_utils
from tests.utils.test_helpers import get_test_config_path


def _get_tpu7x8_topology():
  """Returns virtual `tpu7x-8` topology descriptor across libtpu and TFRT backends."""
  th = accelerator_to_spec_map.get_system_characteristics("tpu7x-8")
  try:
    return get_topology_desc(
        platform=th.platform,
        topology_name=th.topology_name,
        chip_config_name=th.chip_config_name,
        chips_per_host_bounds=th.chips_per_host_bounds,
        num_slices=1,
        wrap=th.wrap,
    )
  except TypeError:
    return get_topology_desc(
        platform=th.platform,
        topology_name=th.topology_name,
        chips_per_host_bounds=th.chips_per_host_bounds,
        num_slices=1,
    )


def _tiny_m3_config(output_dir: str, **overrides):
  """Creates a tiny M3 configuration for testing."""
  args = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "override_model_config": "true",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_decoder_layers": "2",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "2",
      "max_target_length": "16",
      "steps": "4",
      "learning_rate": "1e-3",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_offload_test",
  }
  num_devices = len(jax.devices())
  if num_devices >= 4:
    args["ici_tensor_parallelism"] = "2"
    args["ici_fsdp_parallelism"] = str(num_devices // 2)
  else:
    args["ici_fsdp_parallelism"] = str(num_devices)
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


def _param_paths(model: nnx.Module) -> list[tuple]:
  return [path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))]


def _param_numpy_map(model: nnx.Module) -> dict[tuple, np.ndarray]:
  return {path: np.asarray(var[...]) for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param))}


class ToMemoryTest(parameterized.TestCase):
  """Tests `to_memory` across NNX variables, sharded arrays, tracers, and scalars."""

  def test_preserves_nnx_variable_subclass_and_metadata(self):
    devices = np.array(jax.devices()).reshape(-1)
    mesh = Mesh(devices, ("fsdp",))
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      var = nnx.Param(
          jnp.ones((len(devices) * 4, 8), dtype=jnp.float32),
          out_sharding=P("fsdp", None),
          eager_sharding=True,
          custom_tag="keep_me",
      )
      host_var = m3_offload.to_memory(var, "pinned_host")
      self.assertIsInstance(host_var, nnx.Param)
      self.assertEqual(host_var.get_metadata("out_sharding"), P("fsdp", None))
      self.assertEqual(host_var.get_metadata("custom_tag"), "keep_me")
      self.assertEqual(host_var[...].sharding.memory_kind, "pinned_host")
      self.assertEqual(host_var[...].sharding.spec, P("fsdp", None))

      dev_var = m3_offload.to_memory(host_var, "device")
      self.assertEqual(dev_var[...].sharding.memory_kind, "device")
      self.assertEqual(dev_var[...].sharding.spec, P("fsdp", None))
      np.testing.assert_array_equal(np.asarray(dev_var[...]), np.ones((len(devices) * 4, 8), dtype=np.float32))

  def test_handles_sharded_arrays_shape_dtype_structs_tracers_and_scalars(self):
    dev0 = jax.local_devices()[0]
    single_sharding = SingleDeviceSharding(dev0)
    sds = jax.ShapeDtypeStruct((4, 4), jnp.float32, sharding=single_sharding)
    tree = {
        "arr": jax.device_put(jnp.arange(8, dtype=jnp.float32), single_sharding),
        "sds": sds,
        "sds_plain": jax.ShapeDtypeStruct((2,), jnp.int32),
        "np_arr": np.array([1.0, 2.0], dtype=np.float32),
        "none": None,
        "int": 7,
        "float": 3.5,
        "str": "meta",
    }
    moved = m3_offload.to_memory(tree, "pinned_host")
    self.assertEqual(moved["arr"].sharding.memory_kind, "pinned_host")
    self.assertEqual(moved["sds"].sharding.memory_kind, "pinned_host")
    self.assertIsNone(getattr(moved["sds_plain"], "sharding", None))
    self.assertEqual(moved["np_arr"].sharding.memory_kind, "pinned_host")
    self.assertIsNone(moved["none"])
    self.assertEqual(moved["int"], 7)
    self.assertEqual(moved["float"], 3.5)
    self.assertEqual(moved["str"], "meta")

    @jax.jit
    def roundtrip_traced(x):
      on_host = m3_offload.to_memory(x, "pinned_host")
      return m3_offload.to_memory(on_host, "device") + 1.0

    out = roundtrip_traced(jnp.ones((4,), dtype=jnp.float32))
    self.assertEqual(out.sharding.memory_kind, "device")
    np.testing.assert_allclose(np.asarray(out), np.full((4,), 2.0, dtype=np.float32))

  def test_scalar_0d_device_put_fallback_and_nonscalar_fail_closed(self):
    """Verifies 0D scalar `to_memory(..., 'device')` falls back via NumPy on PJRT layout errors while 1D fails."""
    mesh = Mesh(np.array(jax.local_devices()[:1]), ("data",))
    host_sharding = NamedSharding(mesh, P(), memory_kind="pinned_host")
    scalar_arr = jnp.asarray(7, dtype=jnp.int32)
    scalar_var = nnx.OptVariable(jnp.asarray(9, dtype=jnp.int32))
    orig_device_put = jax.device_put
    array_failures = 0
    numpy_fallbacks = 0

    def _pjrt_cpu_0d_layout_error(val, device=None, **kwargs):
      nonlocal array_failures, numpy_fallbacks
      if isinstance(val, jax.Array) and getattr(val, "ndim", -1) == 0:
        array_failures += 1
        raise jax.errors.JaxRuntimeError(
            "UNIMPLEMENTED: Unsupported layout used for PjRt CPU buffers: {:S(5)} (original) vs. s32[] (canonicalized)"
        )
      if isinstance(val, np.ndarray) and getattr(val, "ndim", -1) == 0:
        numpy_fallbacks += 1
      return orig_device_put(val, device, **kwargs)

    with mock.patch.object(jax, "device_put", side_effect=_pjrt_cpu_0d_layout_error):
      moved_tree = m3_offload.to_memory({"step": scalar_arr, "opt_step": scalar_var}, "device")
      moved_host = m3_offload.to_memory(scalar_arr, "pinned_host")

    self.assertEqual(array_failures, 3)
    self.assertEqual(numpy_fallbacks, 2)
    self.assertEqual(moved_tree["step"].ndim, 0)
    self.assertEqual(moved_tree["step"].dtype, jnp.int32)
    self.assertEqual(int(moved_tree["step"]), 7)
    self.assertEqual(moved_tree["step"].sharding.memory_kind, "device")
    self.assertEqual(int(moved_tree["opt_step"].get_raw_value()), 9)
    self.assertEqual(moved_tree["opt_step"].get_raw_value().sharding.memory_kind, "device")
    self.assertIs(moved_host, scalar_arr)
    self.assertEqual(host_sharding.with_memory_kind("device").memory_kind, "device")

    # Negative control: 1D+ JaxRuntimeError on device_put must NOT be swallowed.
    def _nonscalar_layout_error(val, device=None, **kwargs):
      if isinstance(val, jax.Array):
        raise jax.errors.JaxRuntimeError("UNIMPLEMENTED: 1D layout failure")
      return orig_device_put(val, device, **kwargs)

    with mock.patch.object(jax, "device_put", side_effect=_nonscalar_layout_error):
      with self.assertRaisesRegex(jax.errors.JaxRuntimeError, "1D layout failure"):
        m3_offload.to_memory(jnp.ones((4,), dtype=jnp.float32), "device")

  def test_rejects_unknown_memory_kind(self):
    """Negative control: invalid memory kind must fail loudly."""
    with self.assertRaisesRegex(ValueError, "Unsupported memory kind"):
      m3_offload.to_memory(jnp.ones((2,)), "hbm_magic")


class HostOffloadParityTest(parameterized.TestCase):
  """Verifies path-transparency and step-for-step loss/weight parity across offload modes."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_all_offload_modes_preserve_paths_and_match_step_trajectory(self, scan_layers: bool):
    base_cfg = _tiny_m3_config(
        self.output_dir,
        scan_layers=scan_layers,
        optimizer_memory_host_offload=False,
        parameter_memory_host_offload=False,
    )
    mesh = mesh_from_config(base_cfg)
    sharding = sharding_from_config(base_cfg)
    batches = [
        loop.make_synthetic_batch(base_cfg, mesh, sharding, jax.random.fold_in(jax.random.key(0), s)) for s in range(4)
    ]

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"), nnx.use_eager_sharding(True):
      base_model, base_opt = loop.setup_model(base_cfg, mesh, sharding=sharding)
      base_cls = type(base_model)
      base_paths = _param_paths(base_model)
      base_losses = [float(loop.train_step(base_model, base_opt, b)) for b in batches]
      base_final_params = _param_numpy_map(base_model)

      for opt_offload, param_offload in ((True, False), (False, True), (True, True)):
        cfg = _tiny_m3_config(
            self.output_dir,
            scan_layers=scan_layers,
            optimizer_memory_host_offload=opt_offload,
            parameter_memory_host_offload=param_offload,
        )
        model, opt = loop.setup_model(cfg, mesh, sharding=sharding)
        self.assertIsInstance(model, base_cls)
        self.assertEqual(_param_paths(model), base_paths)

        param_kinds = {
            v[...].sharding.memory_kind
            for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
        }
        expected_param_kind = "pinned_host" if param_offload else "device"
        self.assertEqual(param_kinds, {expected_param_kind})

        opt_arrays = [
            v[...] if isinstance(v, nnx.Variable) else v
            for v in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
            if hasattr(v[...] if isinstance(v, nnx.Variable) else v, "sharding")
        ]
        self.assertNotEmpty(opt_arrays)
        expected_opt_kind = "pinned_host" if opt_offload else "device"
        self.assertTrue(all(a.sharding.memory_kind == expected_opt_kind for a in opt_arrays if getattr(a, "ndim", 0) > 0))

        losses = [float(loop.train_step(model, opt, b)) for b in batches]
        np.testing.assert_allclose(losses, base_losses, rtol=1e-6, atol=1e-6)

        # Verify post-step memory kinds remain parked in their configured spaces.
        post_param_kinds = {
            v[...].sharding.memory_kind
            for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
        }
        self.assertEqual(post_param_kinds, {expected_param_kind})
        post_opt_arrays = [
            v[...] if isinstance(v, nnx.Variable) else v
            for v in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
            if hasattr(v[...] if isinstance(v, nnx.Variable) else v, "sharding")
        ]
        self.assertTrue(
            all(a.sharding.memory_kind == expected_opt_kind for a in post_opt_arrays if getattr(a, "ndim", 0) > 0)
        )

        final_params = _param_numpy_map(model)
        for path, base_arr in base_final_params.items():
          np.testing.assert_allclose(final_params[path], base_arr, rtol=1e-6, atol=1e-6)

      # Negative control: perturbing learning_rate must break trajectory parity.
      perturbed_cfg = _tiny_m3_config(
          self.output_dir,
          scan_layers=scan_layers,
          learning_rate="5e-2",
          optimizer_memory_host_offload=True,
          parameter_memory_host_offload=False,
      )
      perturbed_model, perturbed_opt = loop.setup_model(perturbed_cfg, mesh, sharding=sharding)
      perturbed_losses = [float(loop.train_step(perturbed_model, perturbed_opt, b)) for b in batches]
      self.assertFalse(np.allclose(perturbed_losses[1:], base_losses[1:], rtol=1e-4, atol=1e-4))

  def test_model_creation_utils_wires_parameter_offload_for_train_mode_only(self):
    cfg = _tiny_m3_config(self.output_dir, scan_layers=False, parameter_memory_host_offload=True)
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertIsInstance(train_model, m3_offload.HostOffloadedParameters)
      for var in jax.tree.leaves(nnx.state(train_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "pinned_host")

      batch_size = max(2, mesh.shape["fsdp"])
      tokens = jnp.ones((batch_size, 8), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (batch_size, 8))
      logits = train_model(tokens, decoder_positions=positions)
      self.assertEqual(logits.shape, (batch_size, 8, cfg.vocab_size))
      hidden = jnp.ones((batch_size, 8, cfg.emb_dim), dtype=jnp.float32)
      tiled_logits = train_model.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic=True)
      self.assertEqual(tiled_logits.shape, (batch_size, 8, cfg.vocab_size))
      for var in jax.tree.leaves(nnx.state(train_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "pinned_host")

      prefill_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
      self.assertNotIsInstance(prefill_model, m3_offload.HostOffloadedParameters)
      for var in jax.tree.leaves(nnx.state(prefill_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "device")

  def test_vocab_tiling_hook_raises_cleanly_when_missing_on_parent(self):
    class _MinimalModule(nnx.Module):
      """Minimal module without vocab tiling helper."""

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))

      def __call__(self, x):
        return x @ self.w[...]

    class _Cfg:
      """Minimal config stub enabling parameter host offload."""

      parameter_memory_host_offload = True

    mod = m3_offload.maybe_offload_parameters(_MinimalModule(), _Cfg())
    with self.assertRaisesRegex(AttributeError, "logits_from_hidden_states_for_vocab_tiling"):
      mod.logits_from_hidden_states_for_vocab_tiling(jnp.ones((2, 2)))


class RuntimeHbmTelemetryTest(parameterized.TestCase):
  """Tests `get_device_hbm_stats` and `loop._log_step` HBM reporting."""

  def test_cpu_and_empty_memory_stats_return_none(self):
    self.assertIsNone(m3_offload.get_device_hbm_stats())

    class _NoneStatsDevice:
      """Mock device returning None from memory_stats."""

      def memory_stats(self):
        return None

    class _IncompleteStatsDevice:
      """Mock device omitting bytes_in_use/peak_bytes_in_use."""

      def memory_stats(self):
        return {"bytes_reserved": 1024}

    class _NonAddressablePathwaysDevice:
      """Mock non-addressable Pathways IFRT device."""

      def memory_stats(self):
        raise jax.errors.JaxRuntimeError("INVALID_ARGUMENT: MemoryStats is only supported for addressable PjRt devices.")

    self.assertIsNone(m3_offload.get_device_hbm_stats(_NoneStatsDevice()))
    self.assertIsNone(m3_offload.get_device_hbm_stats(_IncompleteStatsDevice()))
    self.assertIsNone(m3_offload.get_device_hbm_stats(_NonAddressablePathwaysDevice()))

  def test_tpu_memory_stats_captures_reserved_workspace_and_logs_in_step(self):
    gib = 2**30

    # Mirrors DSv4.1 TPU7x runtime observation: `peak_bytes_in_use` reports 48 GiB
    # while XLA holds 46 GiB of HLO temporary workspace in `bytes_reserved` on top
    # of 40 GiB live buffers (`allocated_bytes = 86 GiB`).
    class _MockTpuDevice:
      """Mock TPU device holding reserved HLO workspace."""

      def memory_stats(self):
        return {
            "bytes_in_use": 40 * gib,
            "bytes_reserved": 46 * gib,
            "peak_bytes_in_use": 48 * gib,
        }

    mock_tpu = _MockTpuDevice()
    stats = m3_offload.get_device_hbm_stats(mock_tpu)
    self.assertIsNotNone(stats)
    assert stats is not None
    self.assertEqual(stats["allocated_bytes"], float(86 * gib))
    self.assertEqual(stats["peak_bytes_in_use"], float(48 * gib))
    self.assertEqual(stats["peak_allocated_bytes"], float(86 * gib))
    self.assertAlmostEqual(stats["hbm_allocated_gib"], 86.0)
    self.assertAlmostEqual(stats["hbm_peak_in_use_gib"], 48.0)
    # Negative control: `peak_bytes_in_use` alone misses 38 GiB of reserved HBM.
    self.assertGreater(stats["hbm_allocated_gib"] - stats["hbm_peak_in_use_gib"], 30.0)

    with mock.patch.object(max_logging, "log") as log_spy:
      loop._log_step(0, jnp.asarray(2.5, dtype=jnp.float32), 0.0, 1024, device=mock_tpu)  # pylint: disable=protected-access
      tpu_msg = log_spy.call_args[0][0]
      self.assertIn("hbm_allocated_gib=86.00", tpu_msg)
      self.assertIn("hbm_peak_in_use_gib=48.00", tpu_msg)

      log_spy.reset_mock()
      loop._log_step(1, jnp.asarray(2.4, dtype=jnp.float32), 0.0, 1024)  # pylint: disable=protected-access
      cpu_msg = log_spy.call_args[0][0]
      self.assertNotIn("hbm_allocated_gib", cpu_msg)

    class _ReleasedWorkspaceTpuDevice:
      """Mock TPU device where XLA released reserved workspace before `_log_step`."""

      def memory_stats(self):
        return {
            "bytes_in_use": 2 * gib,
            "bytes_reserved": 4 * gib,
            "peak_bytes_in_use": 4 * gib,
            "peak_allocated_bytes": 86 * gib,
        }

    released_stats = m3_offload.get_device_hbm_stats(_ReleasedWorkspaceTpuDevice())
    assert released_stats is not None
    self.assertEqual(released_stats["allocated_bytes"], float(6 * gib))
    self.assertEqual(released_stats["peak_allocated_bytes"], float(86 * gib))
    self.assertEqual(released_stats["hbm_peak_allocated_gib"], 86.0)
    with mock.patch.object(max_logging, "log") as rel_spy:
      loop._log_step(  # pylint: disable=protected-access
          2, jnp.asarray(2.1, dtype=jnp.float32), 0.0, 1024, device=_ReleasedWorkspaceTpuDevice()
      )
      rel_msg = rel_spy.call_args[0][0]
      self.assertIn("hbm_allocated_gib=6.00", rel_msg)
      self.assertIn("hbm_peak_in_use_gib=4.00", rel_msg)
      self.assertIn("hbm_peak_allocated_gib=86.00", rel_msg)

  def test_memory_report_reexport_matches_utils_module(self):
    self.assertIs(te_memory_report.memory_report, memory_report.memory_report)
    self.assertIs(te_memory_report.format_memory_report, memory_report.format_memory_report)
    self.assertIs(te_memory_report.KernelMemory, memory_report.KernelMemory)

  @pytest.mark.tpu_backend
  def test_host_offloaded_optimizer_init_avoids_device_hbm_spike(self):
    out_dir = self.create_tempdir().full_path
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    cfg = _tiny_m3_config(
        out_dir,
        optimizer_memory_host_offload="true",
        parameter_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    tx = optax.adamw(1e-3)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      device_puts = []
      orig_device_put = jax.device_put

      def _spy_device_put(*args, **kwargs):
        target = args[1] if len(args) > 1 else kwargs.get("device")
        if getattr(target, "memory_kind", None) == "device":
          device_puts.append(target)
        return orig_device_put(*args, **kwargs)

      with mock.patch.object(m3_offload.jax, "device_put", side_effect=_spy_device_put):
        opt = m3_offload.HostOffloadedOptimizer(model, tx, offload_opt_state=True)

    # Neither `param_vars` nor `opt_state` is staged to `"device"` during `HostOffloadedOptimizer.__init__`.
    self.assertEmpty(device_puts)
    opt_leaves = [
        v
        for v in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ]
    self.assertNotEmpty(opt_leaves)
    for leaf in opt_leaves:
      self.assertEqual(leaf.get_metadata("memory_kind"), "pinned_host")
      raw = leaf.get_raw_value()
      if hasattr(raw, "sharding") and getattr(raw, "ndim", 0) > 0:
        self.assertEqual(raw.sharding.memory_kind, "pinned_host")

    # Verify on `tpu7x-8` AOT that `_build_opt_state_out_shardings` compiles `tx.init` directly into `pinned_host`
    # (`host_output_size_in_bytes > 0`, `argument_size_in_bytes == 0`) rather than allocating moments in HBM.
    topo = _get_tpu7x8_topology()
    aot_cfg = _tiny_m3_config(
        out_dir,
        compile_topology="tpu7x-8",
        ici_fsdp_parallelism="4",
        ici_tensor_parallelism="2",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        optimizer_memory_host_offload="true",
        parameter_memory_host_offload="true",
    )
    tpu_mesh = mesh_from_config(aot_cfg, devices=topo.devices)
    with jax.set_mesh(tpu_mesh):
      abs_model = nnx.eval_shape(
          lambda: model_creation_utils.create_model(aot_cfg, tpu_mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      )
      abs_params = nnx.state(abs_model, nnx.Param)
      host_out_sh = m3_offload._build_opt_state_out_shardings(abs_params, tx, "pinned_host")  # pylint: disable=protected-access
      dev_out_sh = m3_offload._build_opt_state_out_shardings(abs_params, tx, "device")  # pylint: disable=protected-access
      ma_host = jax.jit(tx.init, out_shardings=host_out_sh).lower(abs_params).compile().memory_analysis()
      ma_dev = jax.jit(tx.init, out_shardings=dev_out_sh).lower(abs_params).compile().memory_analysis()
    self.assertEqual(ma_host.argument_size_in_bytes, 0)
    self.assertGreater(ma_host.host_output_size_in_bytes, 100_000)
    self.assertLess(ma_host.output_size_in_bytes, 256)
    self.assertGreater(ma_dev.output_size_in_bytes, 100_000)
    self.assertEqual(ma_dev.host_output_size_in_bytes, 0)

  def test_eval_shape_preserves_pinned_host_sharding_for_aot_compile(self):
    out_dir = self.create_tempdir().full_path
    for offload_opt, offload_param in ((False, False), (True, False), (False, True), (True, True)):
      cfg = _tiny_m3_config(
          out_dir,
          optimizer_memory_host_offload=str(offload_opt).lower(),
          parameter_memory_host_offload=str(offload_param).lower(),
      )
      mesh = mesh_from_config(cfg)
      with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
        abs_model, abs_opt = nnx.eval_shape(lambda c=cfg, m=mesh: loop.setup_model(c, m))

      param_vars = [
          v
          for v in jax.tree.leaves(nnx.state(abs_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable)
      ]
      opt_vars = [
          v
          for v in jax.tree.leaves(nnx.state(abs_opt, nnx.OptState), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable) and v.get_raw_value().ndim > 0
      ]
      self.assertNotEmpty(param_vars)
      self.assertNotEmpty(opt_vars)
      for var in param_vars:
        kind = var.get_raw_value().sharding.memory_kind
        if offload_param:
          self.assertEqual(kind, "pinned_host")
        else:
          self.assertIn(kind, (None, "device"))
      for var in opt_vars:
        kind = var.get_raw_value().sharding.memory_kind
        if offload_opt:
          self.assertEqual(kind, "pinned_host")
        elif offload_param:
          self.assertEqual(kind, "device")
        else:
          self.assertIn(kind, (None, "device"))

  @pytest.mark.tpu_backend
  @parameterized.named_parameters(("scanned", True), ("unscanned", False))
  def test_tpu7x_aot_compile_reduces_device_argument_and_kernel_device_peak(self, scan_layers: bool):
    """Verifies `tpu7x-8` AOT compilation reduces device argument bytes under host offload."""
    out_dir = self.create_tempdir().full_path
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    topo = _get_tpu7x8_topology()

    def _compile_step(offload_opt: bool, offload_param: bool = False):
      cfg = _tiny_m3_config(
          out_dir,
          compile_topology="tpu7x-8",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          scan_layers=str(scan_layers).lower(),
          optimizer_memory_host_offload=str(offload_opt).lower(),
          parameter_memory_host_offload=str(offload_param).lower(),
      )
      tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
      sharding = sharding_from_config(cfg)
      with jax.set_mesh(tpu_mesh), nnx.use_eager_sharding(True):
        data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
        data_sharding = NamedSharding(tpu_mesh, data_spec)
        batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
        batch = {
            "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
        }
        abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh))
        lowered = loop.train_step.lower(abs_model, abs_opt, batch)
      compiled = lowered.compile()
      rows = memory_report.memory_report({"update": compiled}, {})
      return compiled.memory_analysis(), rows[0], memory_report.format_memory_report(rows)

    mem_a, row_a, report_a = _compile_step(offload_opt=False, offload_param=False)
    mem_b, row_b, report_b = _compile_step(offload_opt=True, offload_param=False)
    mem_d, row_d, _ = _compile_step(offload_opt=False, offload_param=True)
    _, row_c, _ = _compile_step(offload_opt=True, offload_param=True)

    self.assertEqual(mem_a.host_argument_size_in_bytes, 0)
    self.assertEqual(row_a.host_argument, 0)
    self.assertIn(memory_report.PEAK_LABEL, report_a)
    self.assertIn(memory_report.PEAK_LABEL, report_b)

    offloaded_opt_bytes = row_a.argument - row_b.argument
    self.assertGreater(offloaded_opt_bytes, int(0.6 * row_a.argument))
    self.assertEqual(row_b.host_argument, offloaded_opt_bytes)
    self.assertEqual(mem_b.host_argument_size_in_bytes, offloaded_opt_bytes)
    self.assertLess(row_b.argument, row_a.argument)

    # Parameter-only offload `(False, True)` keeps moments in HBM and moves parameters to S(5) pinned_host.
    offloaded_param_bytes = row_a.argument - row_d.argument
    self.assertGreater(offloaded_param_bytes, 0)
    self.assertEqual(row_d.host_argument, offloaded_param_bytes)
    self.assertEqual(mem_d.host_argument_size_in_bytes, offloaded_param_bytes)
    self.assertLess(row_d.argument, row_a.argument)

    # Offloading both parameters and optimizer state reduces device argument below either alone.
    self.assertLess(row_c.argument, row_b.argument)
    self.assertLess(row_c.argument, row_d.argument)
    self.assertGreater(row_c.host_argument, row_b.host_argument)
    self.assertGreater(row_c.host_argument, row_d.host_argument)

  def _run_legacy_offload_steps(
      self,
      out_dir: str,
      *,
      offload_opt: bool,
      offload_param: bool = False,
      scan_layers: bool = True,
      param_scan_axis: int = 1,
      grad_accum: int = 1,
  ):
    """Runs two compiled `train.py` steps and returns initial/post-step memory kinds and losses."""
    cfg = _tiny_m3_config(
        out_dir,
        dataset_type="synthetic",
        scan_layers=str(scan_layers).lower(),
        param_scan_axis=str(param_scan_axis),
        gradient_accumulation_steps=str(grad_accum),
        optimizer_memory_host_offload=str(offload_opt).lower(),
        parameter_memory_host_offload=str(offload_param).lower(),
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      (
          _,
          _,
          state_mesh_shardings,
          _,
          mesh,
          _,
          data_iterator,
          _,
          _,
          _,
          train_state,
      ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
      params_shardings, state_mesh_shardings = sharding_utils.maybe_update_params_sharding_with_opt(
          cfg, state_mesh_shardings
      )
      jit_model, state = nnx.split(train_state)
      p_train_step, _ = train_utils.jit_train_and_eval_step(
          cfg,
          jit_model,
          mesh,
          state,
          state_mesh_shardings,
          legacy_train.train_step,
          params_shardings=params_shardings if grad_accum == 1 else None,
      )
      batch = next(data_iterator)
      init_opt_kinds = {
          x.sharding.memory_kind
          for x in jax.tree.leaves(state.optimizer)
          if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
      }
      init_param_kinds = {x.sharding.memory_kind for x in jax.tree.leaves(state.model) if hasattr(x, "sharding")}
      state, metrics0 = p_train_step(state, batch)
      step0_opt_kinds = {
          x.sharding.memory_kind
          for x in jax.tree.leaves(state.optimizer)
          if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
      }
      step0_param_kinds = {x.sharding.memory_kind for x in jax.tree.leaves(state.model) if hasattr(x, "sharding")}
      state, metrics1 = p_train_step(state, batch)
    step1_opt_kinds = {
        x.sharding.memory_kind
        for x in jax.tree.leaves(state.optimizer)
        if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
    }
    step1_param_kinds = {x.sharding.memory_kind for x in jax.tree.leaves(state.model) if hasattr(x, "sharding")}
    losses = [float(metrics0["scalar"]["learning/loss"]), float(metrics1["scalar"]["learning/loss"])]
    return (
        init_opt_kinds,
        step0_opt_kinds,
        step1_opt_kinds,
        init_param_kinds,
        step0_param_kinds,
        step1_param_kinds,
        losses,
    )

  def test_legacy_train_step_runtime_keeps_optimizer_in_pinned_host_across_steps(self):
    """Verifies scanned `train.py` steps keep offloaded optimizer and parameter state in `pinned_host`."""
    out_dir = self.create_tempdir().full_path
    # Negative control: offload disabled keeps both optimizer and parameter state on device across steps.
    (
        init_dev,
        step0_dev,
        step1_dev,
        init_p_dev,
        step0_p_dev,
        step1_p_dev,
        losses_dev,
    ) = self._run_legacy_offload_steps(out_dir, offload_opt=False, offload_param=False)
    self.assertEqual(init_dev, {"device"})
    self.assertEqual(step0_dev, {"device"})
    self.assertEqual(step1_dev, {"device"})
    self.assertEqual(init_p_dev, {"device"})
    self.assertEqual(step0_p_dev, {"device"})
    self.assertEqual(step1_p_dev, {"device"})

    for offload_opt, offload_param in ((True, False), (False, True), (True, True)):
      (
          init_o,
          step0_o,
          step1_o,
          init_p,
          step0_p,
          step1_p,
          losses,
      ) = self._run_legacy_offload_steps(out_dir, offload_opt=offload_opt, offload_param=offload_param, param_scan_axis=1)
      exp_o = {"pinned_host"} if offload_opt else {"device"}
      exp_p = {"pinned_host"} if offload_param else {"device"}
      self.assertEqual(init_o, exp_o)
      self.assertEqual(step0_o, exp_o)
      self.assertEqual(step1_o, exp_o)
      self.assertEqual(init_p, exp_p)
      self.assertEqual(step0_p, exp_p)
      self.assertEqual(step1_p, exp_p)
      np.testing.assert_allclose(losses, losses_dev, rtol=1e-6, atol=1e-6)

    _, _, _, _, step0_p_ax0, _, losses_ax0 = self._run_legacy_offload_steps(
        out_dir, offload_opt=True, offload_param=True, scan_layers=True, param_scan_axis=0
    )
    self.assertEqual(step0_p_ax0, {"pinned_host"})
    np.testing.assert_allclose(losses_ax0, losses_dev, rtol=1e-5, atol=1e-5)

  def test_legacy_train_step_unscanned_parameter_offload_parity(self):
    """Verifies unscanned `param_scan_axis=1` parameter host offload matches device losses in `train.py`."""
    out_dir = self.create_tempdir().full_path
    _, _, _, _, step0_p_unscan, _, losses_unscan_off = self._run_legacy_offload_steps(
        out_dir, offload_opt=False, offload_param=False, scan_layers=False, param_scan_axis=1
    )
    _, _, _, _, step0_p_unscan_on, _, losses_unscan_on = self._run_legacy_offload_steps(
        out_dir, offload_opt=False, offload_param=True, scan_layers=False, param_scan_axis=1
    )
    self.assertEqual(step0_p_unscan, {"device"})
    self.assertEqual(step0_p_unscan_on, {"pinned_host"})
    np.testing.assert_allclose(losses_unscan_on, losses_unscan_off, rtol=1e-6, atol=1e-6)

  def test_legacy_train_step_grad_accum_and_classic_scan_axis_guard(self):
    """Verifies `gradient_accumulation_steps=2` parity and classic non-m3 `param_scan_axis != 0` guard."""
    out_dir = self.create_tempdir().full_path
    _, _, _, _, _, _, losses_ga_off = self._run_legacy_offload_steps(
        out_dir, offload_opt=False, offload_param=False, scan_layers=True, param_scan_axis=1, grad_accum=2
    )
    _, _, step1_o_ga, _, step0_p_ga, step1_p_ga, losses_ga_on = self._run_legacy_offload_steps(
        out_dir, offload_opt=True, offload_param=True, scan_layers=True, param_scan_axis=1, grad_accum=2
    )
    self.assertEqual(step1_o_ga, {"pinned_host"})
    self.assertEqual(step0_p_ga, {"pinned_host"})
    self.assertEqual(step1_p_ga, {"pinned_host"})
    np.testing.assert_allclose(losses_ga_on, losses_ga_off, rtol=1e-6, atol=1e-6)

    # Negative control: classic non-m3 `scan_layers=True, param_scan_axis=1, parameter_memory_host_offload=True`
    # still rejects `param_scan_axis != 0` in `get_abstract_state_nnx`.
    classic_cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=default",
            "use_m3_model=false",
            "scan_layers=true",
            "param_scan_axis=1",
            "parameter_memory_host_offload=true",
            "enable_checkpointing=false",
            f"base_output_directory={out_dir}",
            "run_name=classic_param_scan_axis_guard",
        ]
    )
    classic_mesh = Mesh(np.array(jax.devices()).reshape(-1), ("data",))
    with self.assertRaisesRegex(AssertionError, "scan axis 0"):
      maxtext_utils.get_abstract_state_nnx(classic_cfg, classic_mesh, lambda: None, is_training=True)

  @pytest.mark.tpu_backend
  def test_tpu7x_aot_legacy_grad_accum_gathers_params_before_while_loop(self):
    """Verifies GA gathers offloaded parameters before the microbatch `while` loop (`0` `S(5)` ops in loop body)."""
    out_dir = self.create_tempdir().full_path
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    topo_ga = _get_tpu7x8_topology()

    def _compile_legacy_tpu_ga(grad_accum_steps: int) -> str:
      cfg_tpu = _tiny_m3_config(
          out_dir,
          compile_topology="tpu7x-8",
          compile_topology_num_slices="1",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          scan_layers="true",
          param_scan_axis="1",
          gradient_accumulation_steps=str(grad_accum_steps),
          optimizer_memory_host_offload="true",
          parameter_memory_host_offload="true",
      )
      tpu_mesh_ga = mesh_from_config(cfg_tpu, devices=topo_ga.devices)
      shaped_args, shaped_kw, state_shd, _, model_gdef = train_compile.get_shaped_inputs(tpu_mesh_ga, cfg_tpu)
      p_shd, state_shd = sharding_utils.maybe_update_params_sharding_with_opt(cfg_tpu, state_shd)
      in_state_shd = sharding_utils.build_zero1_input_state_mesh_shardings(cfg_tpu, state_shd, p_shd)
      d_shd = sharding_utils.get_input_data_sharding(cfg_tpu, tpu_mesh_ga)
      fn_to_comp, in_shd, out_shd, static_argnums, donate_argnums = maxtext_utils.get_functional_train_with_signature(
          legacy_train.train_step,
          d_shd,
          in_state_shd,
          model_gdef,
          cfg_tpu,
          p_shd,
      )
      compiled = train_compile.jit_and_compile(
          fn_to_comp,
          shaped_args,
          shaped_kw,
          tpu_mesh_ga,
          in_shd,
          out_shd,
          static_argnums,
          donate_argnums,
          cfg_tpu,
          train_compile.axis_rules(cfg_tpu.logical_axis_rules),
      )
      return compiled.as_text() or ""

    hlo_ga1 = _compile_legacy_tpu_ga(1)
    hlo_ga2 = _compile_legacy_tpu_ga(2)
    s5_lines_ga1 = sum(1 for ln in hlo_ga1.splitlines() if "S(5)" in ln)
    s5_lines_ga2 = sum(1 for ln in hlo_ga2.splitlines() if "S(5)" in ln)
    self.assertGreater(s5_lines_ga1, 0)
    self.assertEqual(s5_lines_ga2, s5_lines_ga1)

    comp_map: dict[str, list[str]] = {}
    cur_comp: str | None = None
    for ln in hlo_ga2.splitlines():
      m_hdr = re.match(r"^(?:ENTRY\s+)?(%[\w.\-]+)\s*\(", ln.strip())
      if m_hdr and ln.rstrip().endswith("{"):
        cur_comp = m_hdr.group(1)
        comp_map[cur_comp] = []
      elif cur_comp is not None and ln.startswith("}"):
        cur_comp = None
      elif cur_comp is not None:
        comp_map[cur_comp].append(ln)

    while_roots: set[str] = set()
    for lines in comp_map.values():
      for ln in lines:
        if "while(" in ln:
          while_roots.update(re.findall(r"\bbody=(%[\w.\-]+)", ln))
    visited_while: set[str] = set()
    stack_while = list(while_roots)
    while stack_while:
      cname = stack_while.pop()
      if cname in visited_while or cname not in comp_map:
        continue
      visited_while.add(cname)
      for ln in comp_map[cname]:
        for callee in re.findall(r"\b(?:calls|to_apply|body|true_computation|false_computation)=(%[\w.\-]+)", ln):
          if callee not in visited_while:
            stack_while.append(callee)
    for cname in visited_while:
      self.assertTrue(all("S(5)" not in ln for ln in comp_map[cname]), f"Unexpected S(5) in while body {cname}")

  def test_offloaded_model_class_vocab_tiling_attribute(self):  # pylint: disable=protected-access
    """Verifies `logits_from_hidden_states_for_vocab_tiling` is only present when base class defines it."""

    class _BaseWithoutTiling(nnx.Module):

      def __call__(self, x):
        return x

    class _BaseWithTiling(nnx.Module):

      def __call__(self, x):
        return x

      def logits_from_hidden_states_for_vocab_tiling(self, x):
        return x * 2

    offloaded_cls_without = m3_offload._offloaded_model_class(_BaseWithoutTiling)  # pylint: disable=protected-access
    offloaded_cls_with = m3_offload._offloaded_model_class(_BaseWithTiling)  # pylint: disable=protected-access
    self.assertFalse(hasattr(offloaded_cls_without, "logits_from_hidden_states_for_vocab_tiling"))
    self.assertTrue(hasattr(offloaded_cls_with, "logits_from_hidden_states_for_vocab_tiling"))

  @parameterized.named_parameters(
      ("scanned_none", True, "none"),
      ("scanned_full", True, "full"),
      ("unscanned_none", False, "none"),
      ("unscanned_full", False, "full"),
  )
  def test_create_model_preserves_qwix_quantization_with_parameter_offload(self, scan_layers: bool, remat_pol: str):
    """Verifies `model_creation_utils.create_model` keeps Qwix quantization when parameter offload is enabled."""
    toks = jnp.ones((2, 16), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
    cfg = _tiny_m3_config(
        self.create_tempdir().full_path,
        scan_layers=str(scan_layers).lower(),
        remat_policy=remat_pol,
        quantization="int8",
        use_qwix_quantization="true",
        parameter_memory_host_offload="true",
        optimizer_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertIsInstance(model, m3_offload.HostOffloadedParameters)
      self.assertTrue(hasattr(type(model), "_unquantized_type"))
      param_kinds = {
          v.get_raw_value().sharding.memory_kind
          for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable) and hasattr(v.get_raw_value(), "sharding")
      }
      self.assertEqual(param_kinds, {"pinned_host"})
      self.assertGreater(m3_quant.quantized_matmuls(model, jnp.int8, toks, pos), 0)
      out = model(toks, pos)
      self.assertEqual(out.shape, (2, 16, 256))

  @pytest.mark.tpu_backend
  def test_tpu7x_explicit_sharding_aot_compile_and_zero_hook_warnings(self):
    """Verifies `shard_mode='explicit'` AOT compilation, zero hook warnings, and non-mutating cloning."""
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path

    # `maybe_offload_parameters` returns a `HostOffloadedParameters` clone without mutating `orig.__class__`.
    clone_cfg = _tiny_m3_config(out_dir, scan_layers="false", parameter_memory_host_offload="true")
    cpu_mesh = mesh_from_config(clone_cfg)
    with jax.set_mesh(cpu_mesh), nnx.use_eager_sharding(True):
      orig_cls_cfg = _tiny_m3_config(out_dir, scan_layers="false", parameter_memory_host_offload="false")
      raw_model = model_creation_utils.create_model(orig_cls_cfg, cpu_mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      orig_cls = type(raw_model)
      # Standalone m3 ModelConfig objects omit MaxTextConfig host-offload fields.
      self.assertIs(m3_offload.maybe_offload_parameters(raw_model, raw_model.config), raw_model)
      cloned = m3_offload.maybe_offload_parameters(raw_model, clone_cfg)
      self.assertIs(type(raw_model), orig_cls)
      self.assertNotIsInstance(raw_model, m3_offload.HostOffloadedParameters)
      self.assertIsInstance(cloned, m3_offload.HostOffloadedParameters)
      self.assertEqual(_param_paths(cloned), _param_paths(raw_model))

    topo = _get_tpu7x8_topology()

    def _compile_explicit(offload_opt: bool, offload_param: bool):
      cfg = _tiny_m3_config(
          out_dir,
          shard_mode="explicit",
          compile_topology="tpu7x-8",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          optimizer_memory_host_offload=str(offload_opt).lower(),
          parameter_memory_host_offload=str(offload_param).lower(),
      )
      tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
      sharding = sharding_from_config(cfg)
      with jax.set_mesh(tpu_mesh), nnx.use_eager_sharding(True):
        data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
        data_sharding = NamedSharding(tpu_mesh, data_spec)
        batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
        batch = {
            "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
        }
        abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh))
        compiled = loop.train_step.lower(abs_model, abs_opt, batch).compile()
      row = memory_report.memory_report({"update": compiled}, {})[0]
      param_vars = [
          v
          for v in jax.tree.leaves(nnx.state(abs_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable)
      ]
      opt_vars = [
          v
          for v in jax.tree.leaves(abs_opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable)
      ]
      return row, param_vars, opt_vars

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      row_00, p_00, o_00 = _compile_explicit(False, False)
      row_10, p_10, o_10 = _compile_explicit(True, False)
      row_01, p_01, o_01 = _compile_explicit(False, True)
      row_11, p_11, o_11 = _compile_explicit(True, True)

    hook_warnings = [w for w in caught if "Variable hooks are deprecated" in str(w.message)]
    self.assertEmpty(hook_warnings)

    self.assertTrue(all(v.get_raw_value().sharding.memory_kind in (None, "device") for v in p_00))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind in (None, "device") for v in o_00))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind in (None, "device") for v in p_10))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind == "pinned_host" for v in o_10))
    self.assertTrue(all(isinstance(v, m3_offload.HostOffloadedParam) for v in p_01))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind == "pinned_host" for v in p_01))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind == "device" for v in o_01))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind == "pinned_host" for v in p_11))
    self.assertTrue(all(v.get_raw_value().sharding.memory_kind == "pinned_host" for v in o_11))

    self.assertEqual(row_00.host_argument, 0)
    for row in (row_10, row_01, row_11):
      self.assertEqual(row.argument + row.host_argument, row_00.argument)
      self.assertGreater(row.host_argument, 0)
      self.assertEqual(row.host_alias, row.host_argument)
      self.assertEqual(row.host_output, row.host_argument)
    self.assertLess(row_11.argument, row_10.argument)
    self.assertLess(row_11.argument, row_01.argument)

  @pytest.mark.tpu_backend
  def test_train_compile_main_logs_memory_report_peak(self):
    """Verifies `train_compile.main` emits `memory_report.format_memory_report` output under host offload."""
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path
    prev_prng = jax.config.jax_default_prng_impl
    prev_libtpu = os.environ.get("LIBTPU_INIT_ARGS")
    buf = io.StringIO()
    try:
      with contextlib.redirect_stdout(buf):
        train_compile.main(
            [
                "",
                get_test_config_path(),
                "model_name=qwen3-0.6b",
                "use_m3_model=true",
                "override_model_config=true",
                "base_emb_dim=512",
                "base_mlp_dim=2048",
                "base_num_decoder_layers=2",
                "base_num_query_heads=8",
                "base_num_kv_heads=4",
                "head_dim=64",
                "vocab_size=8192",
                "per_device_batch_size=2",
                "max_target_length=16",
                "compile_topology=tpu7x-8",
                "compile_topology_num_slices=1",
                "ici_fsdp_parallelism=4",
                "ici_tensor_parallelism=2",
                "optimizer_memory_host_offload=true",
                "parameter_memory_host_offload=true",
                "enable_checkpointing=false",
                f"base_output_directory={out_dir}",
                "run_name=train_compile_offload_report",
            ]
        )
    finally:
      jax.config.update("jax_default_prng_impl", prev_prng)
      if prev_libtpu is None:
        os.environ.pop("LIBTPU_INIT_ARGS", None)
      else:
        os.environ["LIBTPU_INIT_ARGS"] = prev_libtpu

    output = buf.getvalue()
    self.assertIn(memory_report.PEAK_LABEL, output)
    self.assertIn("host arg", output)
    self.assertIn("host out", output)
    self.assertIn("host alias", output)
    update_lines = [ln.split() for ln in output.splitlines() if ln.strip().startswith("update ")]
    self.assertLen(update_lines, 1)
    # Columns: kernel(0) arg(1) out(2) alias(3) temp(4) resident(5) +state(6) total(7) host_arg(8) host_out(9) host_alias(10)
    self.assertGreater(float(update_lines[0][8]), 0.0)
    self.assertEqual(update_lines[0][8], update_lines[0][9])
    self.assertEqual(update_lines[0][8], update_lines[0][10])

  def test_host_offloaded_optimizer_aligns_saved_eager_past_non_opt_variable_leaves(self):
    """Verifies per-leaf saved `eager_sharding` stays aligned even if non-`OptVariable` leaves precede `OptVariable`."""

    class _SingleParam(nnx.Module):
      """Minimal module carrying one `eager_sharding=False` parameter."""

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32), eager_sharding=False)

    orig_to_opt = nnx.training.optimizer.to_opt_state

    def _to_opt_preserving_opt_state(tree):
      return jax.tree.map(
          lambda x: x if isinstance(x, nnx.OptState) else orig_to_opt(x),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )

    def _init_with_leading_opt_state(params):
      leading = nnx.OptState(jnp.zeros((), dtype=jnp.int32), eager_sharding=True)
      moments = jax.tree.map(
          lambda v: v.from_metadata(jnp.zeros_like(v.get_raw_value()), v.get_metadata()),
          params,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return (leading, moments)

    tx = optax.GradientTransformation(_init_with_leading_opt_state, lambda u, s, p=None: (u, s))
    with mock.patch.object(nnx.training.optimizer, "to_opt_state", _to_opt_preserving_opt_state):
      opt = m3_offload.HostOffloadedOptimizer(_SingleParam(), tx, wrt=nnx.Param, offload_opt_state=False)
    opt_vars = [
        leaf
        for leaf in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(leaf, nnx.training.optimizer.OptVariable)
    ]
    self.assertLen(opt_vars, 1)
    self.assertFalse(bool(opt_vars[0].eager_sharding))
    self.assertNotIn("_offload_saved_eager", opt_vars[0].get_metadata())

  def test_filter_shardings_for_tree_defaults_missing_paths_to_device_space(self):
    """Verifies _filter_shardings_for_tree maps missing paths to jax.memory.Space.Device."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[: min(2, jax.device_count())]), ("fsdp",))
    dev_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", None))
    host_sharding = dev_sharding.with_memory_kind("pinned_host")
    host_arr = jax.device_put(jnp.ones((4, 4), dtype=jnp.float32), host_sharding)
    self.assertEqual(host_arr.sharding.memory_kind, "pinned_host")

    with jax.set_mesh(mesh):
      target_tree = {
          "trainable": nnx.Param(host_arr, out_sharding=P("fsdp", None), eager_sharding=False),
          "frozen": m3_offload.HostOffloadedParam(
              host_arr, out_sharding=P("fsdp", None), memory_kind="pinned_host", eager_sharding=False
          ),
          "raw_frozen": host_arr,
      }
      # Partial shardings_tree containing only 'trainable' (simulates LoRA trainable-only params_shardings).
      partial_shardings = {"trainable": nnx.Param(dev_sharding, eager_sharding=False)}
      filtered = legacy_train._filter_shardings_for_tree(partial_shardings, target_tree)  # pylint: disable=protected-access

    self.assertEqual(filtered["trainable"].get_raw_value(), dev_sharding)
    self.assertEqual(filtered["frozen"].get_raw_value(), jax.memory.Space.Device)
    self.assertEqual(filtered["raw_frozen"], jax.memory.Space.Device)

    # Passing `filtered` directly to `jax.device_put` inside `jax.jit` (matching `train.py`) moves all leaves to device.
    placed = jax.jit(lambda t: jax.device_put(t, filtered))(target_tree)
    self.assertEqual(placed["frozen"].get_raw_value().sharding.memory_kind, "device")
    if mesh.size > 1:
      self.assertEqual(placed["frozen"].get_raw_value().sharding.spec[0], "fsdp")
      self.assertTrue(placed["frozen"].get_raw_value().sharding.is_equivalent_to(dev_sharding, ndim=2))
    self.assertEqual(placed["raw_frozen"].sharding.memory_kind, "device")
    # Ensure device matmul compiles and runs without pinned_host memory_space mismatch.
    prod = jax.jit(jnp.matmul)(placed["trainable"].get_raw_value(), placed["frozen"].get_raw_value())
    np.testing.assert_allclose(np.asarray(prod), np.full((4, 4), 4.0, dtype=np.float32))

  def test_multimodal_freeze_vision_encoder_params_masks_adamw_weight_decay_and_moments(self):
    """Verifies AdamW masks frozen multimodal `*VisionEncoder*` weights to `set_to_zero()` without `mu`/`nu` arrays."""
    from maxtext.optimizers import optimizers as mt_optimizers  # pylint: disable=import-outside-toplevel

    class _TinyVisionPair(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.Qwen3VLVisionEncoder_0 = nnx.Linear(4, 4, rngs=rngs)  # pylint: disable=invalid-name
        self.Qwen3VLVisionProjector_0 = nnx.Linear(4, 4, rngs=rngs)  # pylint: disable=invalid-name

      def __call__(self, x: jax.Array) -> jax.Array:
        return self.Qwen3VLVisionProjector_0(jax.nn.silu(self.Qwen3VLVisionEncoder_0(x)))

    out_dir = self.create_tempdir().full_path
    x = jnp.ones((2, 4), dtype=jnp.float32)

    def _run_step(freeze_encoder: bool):
      cfg = _tiny_m3_config(
          out_dir,
          model_name="qwen3-vl-2b",
          use_multimodal="true",
          scan_layers="false",
          freeze_vision_encoder_params=str(freeze_encoder).lower(),
          learning_rate="1e-2",
          adam_weight_decay="0.1",
      )
      model = _TinyVisionPair(nnx.Rngs(0))
      before = {tuple(p): np.asarray(v[...]).copy() for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}
      tx = mt_optimizers.get_optimizer(cfg, optax.constant_schedule(1e-2))
      opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
      _, raw_grads = nnx.value_and_grad(lambda m: jnp.sum(m(x) ** 2))(model)
      if freeze_encoder:
        # Simulate stop_gradient on frozen vision encoder: gradient is zero, so only decoupled weight_decay could mutate it.
        flat_g = dict(nnx.to_flat_state(raw_grads))
        for p, g_var in list(flat_g.items()):
          if p[0] == "Qwen3VLVisionEncoder_0":
            flat_g[p] = g_var.replace(jnp.zeros_like(g_var[...]))
        raw_grads = nnx.from_flat_state(flat_g)
      opt.update(model, raw_grads)
      after = {tuple(p): np.asarray(v[...]).copy() for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}
      enc_delta = max(float(np.max(np.abs(after[p] - before[p]))) for p in before if p[0] == "Qwen3VLVisionEncoder_0")
      proj_delta = max(float(np.max(np.abs(after[p] - before[p]))) for p in before if p[0] == "Qwen3VLVisionProjector_0")
      opt_paths = [
          tuple(str(getattr(k, "key", getattr(k, "name", k))) for k in p)
          for p, leaf in jax.tree_util.tree_leaves_with_path(opt.opt_state, is_leaf=lambda v: isinstance(v, nnx.Variable))
          if isinstance(leaf.get_raw_value(), jax.Array)
      ]
      has_enc_moments = any("Qwen3VLVisionEncoder_0" in p for p in opt_paths)
      return enc_delta, proj_delta, has_enc_moments

    frz_enc_d, frz_proj_d, frz_has_enc_mom = _run_step(True)
    unfrz_enc_d, unfrz_proj_d, unfrz_has_enc_mom = _run_step(False)
    self.assertEqual(frz_enc_d, 0.0)
    self.assertGreater(frz_proj_d, 0.0)
    self.assertFalse(frz_has_enc_mom)
    self.assertGreater(unfrz_enc_d, 0.0)
    self.assertGreater(unfrz_proj_d, 0.0)
    self.assertTrue(unfrz_has_enc_mom)


if __name__ == "__main__":
  absltest.main()

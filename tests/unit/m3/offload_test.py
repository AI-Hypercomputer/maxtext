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
from jax.sharding import AxisType, Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.common import train_state_nnx
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


def _qwen38_mini_config(output_dir: str, **overrides):
  """Creates a compact `qwen3.8-flash-next-mini` M3 configuration for selective offload tests."""
  args = {
      "model_name": "qwen3.8-flash-next-mini",
      "override_model_config": "true",
      "use_m3_model": "true",
      "per_device_batch_size": "1",
      "max_target_length": "16",
      "steps": "4",
      "learning_rate": "1e-3",
      "gradient_clipping_threshold": "1.0",
      "adam_weight_decay": "0.1",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_selective_offload_test",
  }
  num_devices = len(jax.devices())
  if num_devices >= 4:
    args["ici_fsdp_parallelism"] = "2"
    args["ici_expert_parallelism"] = str(num_devices // 2)
  else:
    args["ici_fsdp_parallelism"] = str(num_devices)
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


_HLO_DEBUG_SECTIONS = frozenset({"FileNames", "FunctionNames", "FileLocations", "StackFrames"})
_HLO_COMP_DEF_RE = re.compile(r"^(ENTRY\s+)?(%[\w.\-]+)\s*\((.*)\)\s*->\s*(.*\{\s*)$")
_HLO_PARAM_DECL_RE = re.compile(r"(?:^|(?<=,\s)|(?<=\*/))([A-Za-z_][\w.\-]*)(\s*:)")
_HLO_STR_OR_IDENT_RE = re.compile(r'"[^"]*"|%([A-Za-z_][\w.\-]*)')


def _hlo_comp_stem(name: str) -> str:
  stem = re.sub(r"\.\d+(?=\.|_spmd\b|$)", "", name)
  return re.sub(r"\bregion_\d+", "region", stem)


def _hlo_ssa_stem(name: str) -> str:
  return re.sub(r"\.\d+(?=\.|$)", "", name)


def _normalize_hlo(hlo_text: str) -> str:
  """Canonicalizes XLA debug tables and monotonic SSA/subcomputation counters while preserving graph wiring."""
  lines = hlo_text.splitlines()
  comp_map: dict[str, str] = {}
  comp_counts: dict[str, int] = {}
  for line in lines:
    m = _HLO_COMP_DEF_RE.match(line.strip())
    if m:
      raw_comp = m.group(2)[1:]
      if raw_comp not in comp_map:
        stem = _hlo_comp_stem(raw_comp)
        idx = comp_counts.get(stem, 0)
        comp_counts[stem] = idx + 1
        comp_map[raw_comp] = f"{stem}#{idx}"

  out_lines: list[str] = []
  in_debug_section = False
  in_comp = False
  local_map: dict[str, str] = {}
  local_counts: dict[str, int] = {}

  def _map_local(raw_name: str) -> str:
    if raw_name in comp_map:
      return comp_map[raw_name]
    if raw_name not in local_map:
      stem = _hlo_ssa_stem(raw_name)
      idx = local_counts.get(stem, 0)
      local_counts[stem] = idx + 1
      local_map[raw_name] = f"{stem}#{idx}"
    return local_map[raw_name]

  def _replace_param(pm: re.Match[str]) -> str:
    return f"{_map_local(pm.group(1))}{pm.group(2)}"

  def _replace_ident(im: re.Match[str]) -> str:
    return f"%{_map_local(im.group(1))}" if im.group(1) is not None else im.group(0)

  for line in lines:
    stripped = line.strip()
    if not in_comp:
      if stripped in _HLO_DEBUG_SECTIONS:
        in_debug_section = True
        continue
      if in_debug_section:
        if not stripped or re.match(r"^\d+\s+", stripped):
          continue
        in_debug_section = False
      if stripped.startswith("HloModule "):
        line = re.sub(r"^(\s*HloModule\s+)\S+?,", r"\1<module>,", line, count=1)
        out_lines.append(line)
        continue

    m = _HLO_COMP_DEF_RE.match(stripped)
    if m:
      in_comp = True
      local_map.clear()
      local_counts.clear()
      entry_prefix = m.group(1) or ""
      comp_canon = comp_map[m.group(2)[1:]]
      params_str = _HLO_PARAM_DECL_RE.sub(_replace_param, m.group(3))
      out_lines.append(f"{entry_prefix}%{comp_canon} ({params_str}) -> {m.group(4).rstrip()}")
      continue

    if in_comp and (stripped == "}" or stripped.startswith("},")):
      in_comp = False
      out_lines.append(line)
      continue

    if in_comp:
      line = re.sub(r"\s*\bstack_frame_id=\d+", "", line)
      line = re.sub(r",?\s*metadata=\{[^}]*\}", "", line)
      line = _HLO_STR_OR_IDENT_RE.sub(_replace_ident, line)
    if stripped or (out_lines and out_lines[-1] != ""):
      out_lines.append(line)

  return "\n".join(out_lines)


class SelectiveHostOffloadTest(parameterized.TestCase):
  """Tests pattern-selected pinned-host offload for large embedding tables (e.g. Qwen3.8 PLE)."""

  _tpu7x_ple_unsharded_off_cache: dict[str, tuple[int, int, int]] = {}
  _tpu7x_engine_ple_off_cache: dict[str, tuple[int, int]] = {}
  _mtp_ref_cache: dict[int, tuple] = {}

  def setUp(self):
    super().setUp()
    self.enterContext(nnx.use_eager_sharding(True))

  def test_device_host_compute_compat_across_jax_versions(self):
    """Verifies `_device_host_compute` handles `compute_on`, `compute_on2` (`jax 0.11.0`), and context fallback."""
    # pylint: disable=protected-access
    import types  # pylint: disable=import-outside-toplevel

    @contextlib.contextmanager
    def _legacy_cm(compute_type: str):  # pylint: disable=unused-argument
      yield

    real_dec = m3_offload._COMPUTE_ON_DECORATOR
    self.assertIsNotNone(real_dec)
    mod_0110 = types.SimpleNamespace(compute_on=_legacy_cm, compute_on2=real_dec)
    self.assertIs(m3_offload._resolve_compute_on_decorator(mod_0110), real_dec)
    mod_legacy = types.SimpleNamespace(compute_on=_legacy_cm)
    self.assertIsNone(m3_offload._resolve_compute_on_decorator(mod_legacy))

    @m3_offload._device_host_compute(
        out_memory_spaces=(jax.memory.Space.Device, jax.memory.Space.Host),
        compute_on_mod=mod_0110,
    )
    def _pair_0110(x_h: jax.Array):
      doubled = x_h * 2.0
      return doubled, doubled + 1.0

    @m3_offload._device_host_compute(
        out_memory_spaces=(jax.memory.Space.Device, jax.memory.Space.Host),
        compute_on_mod=mod_legacy,
    )
    def _pair_legacy(x_d: jax.Array):
      doubled = x_d * 2.0
      return doubled, doubled + 1.0

    dev0 = jax.local_devices()[0]
    x_host = jax.device_put(
        jnp.array([1.0, 3.0], dtype=jnp.float32),
        SingleDeviceSharding(dev0, memory_kind="pinned_host"),
    )
    y_dev_0110, y_host_0110 = jax.jit(_pair_0110)(x_host)
    self.assertEqual(y_dev_0110.sharding.memory_kind, "device")
    self.assertEqual(y_host_0110.sharding.memory_kind, "pinned_host")
    np.testing.assert_allclose(np.asarray(y_dev_0110), np.array([2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(y_host_0110), np.array([3.0, 7.0], dtype=np.float32))

    x_dev = jax.device_put(jnp.array([1.0, 3.0], dtype=jnp.float32), SingleDeviceSharding(dev0))
    y_dev_leg, y_host_leg = jax.jit(_pair_legacy)(x_dev)
    self.assertEqual(y_dev_leg.sharding.memory_kind, "device")
    self.assertEqual(y_host_leg.sharding.memory_kind, "pinned_host")
    np.testing.assert_allclose(np.asarray(y_dev_leg), np.array([2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(y_host_leg), np.array([3.0, 7.0], dtype=np.float32))

  def test_validation_fails_closed(self):
    out_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, "host_offload_param_patterns"):
      _qwen38_mini_config(out_dir, selective_host_offload="true", host_offload_param_patterns="[]")

    with self.assertRaisesRegex(ValueError, "parameter_memory_host_offload"):
      _qwen38_mini_config(
          out_dir,
          selective_host_offload="true",
          parameter_memory_host_offload="true",
      )

    cfg_unmatched = _qwen38_mini_config(
        out_dir,
        selective_host_offload="true",
        host_offload_param_patterns='["nonexistent_param_path_xyz$"]',
    )
    mesh = mesh_from_config(cfg_unmatched)
    with jax.set_mesh(mesh):
      with self.assertRaisesRegex(ValueError, "matched 0 parameters"):
        loop.setup_model(cfg_unmatched, mesh)

    with self.assertRaises(ValueError):
      _qwen38_mini_config(
          out_dir,
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="unknown_mode",
      )

    with self.assertRaisesRegex(ValueError, "requires use_m3_model=True"):
      pyconfig.initialize(
          ["", get_test_config_path()],
          base_output_directory=out_dir,
          run_name="legacy_selective_offload_test",
          skip_jax_distributed_system=True,
          use_m3_model=False,
          selective_host_offload=True,
          host_offload_param_patterns=["(?:^|/)embedding$"],
      )

    from maxtext.configs import types as cfg_types  # pylint: disable=import-outside-toplevel
    from maxtext.m3.configs.config import TrainerConfig  # pylint: disable=import-outside-toplevel

    with self.assertRaisesRegex(ValueError, "cannot be combined with parameter_memory_host_offload"):
      TrainerConfig(
          selective_host_offload=True,
          parameter_memory_host_offload=True,
          host_offload_param_patterns=["(?:^|/)ple_embedding/ngram_embedding/embedding$"],
      )
    with self.assertRaisesRegex(ValueError, "requires non-empty host_offload_param_patterns"):
      TrainerConfig(selective_host_offload=True)
    with self.assertRaisesRegex(ValueError, "not supported with opt_type='muon'"):
      _qwen38_mini_config(out_dir, selective_host_offload="true", opt_type="muon")
    with self.assertRaisesRegex(ValueError, "not supported with opt_type='muon'"):
      TrainerConfig(
          selective_host_offload=True,
          host_offload_param_patterns=["(?:^|/)ple_embedding/ngram_embedding/embedding$"],
          opt_type="muon",
      )
    with self.assertRaisesRegex(ValueError, "requires non-empty host_offload_param_patterns"):
      pyconfig.initialize_pydantic(
          [
              "",
              get_test_config_path("post_train/rl.yml"),
              "selective_host_offload=true",
              "host_offload_param_patterns=[]",
          ],
          config_class=cfg_types.RLConfig,
          enable_checkpointing=False,
          run_name="rl_selective_offload_test",
      )
    with self.assertRaisesRegex(
        ValueError,
        "selective_host_offload=True is not supported on the post-training RL path",
    ):
      cfg_types.RLConfig(
          model_name="qwen3-8b",
          tokenizer_path="src/maxtext/assets/tokenizers/tokenizer.llama2",
          selective_host_offload=True,
          host_offload_param_patterns=["(?:^|/)embedding$"],
      )

    class _DictTupleMod(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.embeds = nnx.Dict({"ple": nnx.Embed(16, 8, rngs=rngs)})
        self.embed_tuple = nnx.data((self.embeds["ple"],))

    dt_mod = _DictTupleMod(nnx.Rngs(0))
    dt_cfg = TrainerConfig(
        selective_host_offload=True,
        host_offload_param_patterns=[r"(?:embeds/ple|embed_tuple/0)/embedding$"],
    )
    dt_off = m3_offload.maybe_offload_parameters(dt_mod, dt_cfg)
    self.assertIsInstance(dt_off.embeds["ple"], m3_offload.HostOffloadedEmbed)
    self.assertIs(dt_off.embed_tuple[0], dt_off.embeds["ple"])

  def test_selective_offload_fails_closed_instead_of_silent_fallback(self):
    """Verifies missing tokens, pre-gather shape mismatch, and dense lazy_sparse raise ValueError."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    cfg = _qwen38_mini_config(out_dir, scan_layers="true", selective_host_offload="true")
    mesh = mesh_from_config(cfg)
    sharding = sharding_from_config(cfg)

    with jax.set_mesh(mesh):
      model, _ = loop.setup_model(cfg, mesh, sharding)
      bad_batch = {"targets": jnp.zeros((4, 16), dtype=jnp.int32)}

      # 1. value_and_grad / pure_value_and_grad / engine_value_and_grad refuse dense fallback when tokens=None.
      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.value_and_grad(loop.loss_fn, model, bad_batch)

      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.pure_value_and_grad(
            lambda p, cp, r, c, d: (jnp.array(0.0), None),
            graphdef,
            params,
            custom_params=nnx.State({}),
            rest=rest,
            config=cfg,
            data=bad_batch,
        )

      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.engine_value_and_grad(
            lambda p, r, b: (jnp.array(0.0), (None, r)),
            graphdef,
            params,
            rest,
            bad_batch,
        )

      # 2. HostOffloadedEmbed refuses pre-gather shape mismatch instead of bypassing pre_gathered.
      ple_embeds = m3_offload._collect_host_embeds(model)  # pylint: disable=protected-access
      self.assertEqual(len(ple_embeds), 1)
      ple_path, ple_embed, _ = ple_embeds[0]
      dummy_pre_ids = jnp.zeros((2, 4), dtype=jnp.int32)
      dummy_pre_vals = jnp.zeros((2, 4, ple_embed.features), dtype=jnp.float32)
      with m3_offload._pregathered_context({ple_path: (dummy_pre_ids, dummy_pre_vals)}):  # pylint: disable=protected-access
        with self.assertRaisesRegex(ValueError, "pre-gather shape mismatch"):
          ple_embed(jnp.zeros((2, 5), dtype=jnp.int32))

      # 3. lazy_sparse rejects dense host gradients both at config/make_optimizer time and in update().
      cfg_ls = _qwen38_mini_config(
          out_dir,
          scan_layers="true",
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="lazy_sparse",
      )
      with self.assertRaisesRegex(ValueError, "incompatible with dense pinned_host gradients"):
        m3_offload.make_optimizer(model, optax.adamw(1e-3), cfg_ls, dense_host_grads=True)

      opt_ls = m3_offload.make_optimizer(model, optax.adamw(1e-3), cfg_ls)
      dense_grads = jax.tree.map(
          lambda v: (
              v.replace(
                  m3_offload.to_memory(jnp.zeros_like(v.get_raw_value()), "pinned_host")
                  if m3_offload._is_host_leaf(v)  # pylint: disable=protected-access
                  else jnp.zeros_like(v.get_raw_value())
              )
              if isinstance(v, nnx.Variable)
              else jnp.zeros_like(v)
          ),
          nnx.state(model, nnx.Param),
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      with self.assertRaisesRegex(ValueError, "requires SparseHostGrad"):
        opt_ls.update(model, dense_grads)

    # 4. AbstractMaxTextEngine fails closed at construction time for lazy_sparse (when tunix is installed).
    if importlib.util.find_spec("tunix") is not None:
      from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=import-outside-toplevel

      cfg_eng_ls = _qwen38_mini_config(
          out_dir,
          scan_layers="true",
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="lazy_sparse",
      )
      with self.assertRaisesRegex(ValueError, "incompatible with dense pinned_host gradients"):
        maxtext_engine_compile.AbstractMaxTextEngine(cfg_eng_ls, mesh)

  def test_sharded_2d_embed_gather_sqnorm_clip_and_optimizer_modes(self):
    """Tests 2-axis row-sharded P(('fsdp', 'ep'), None) table with non-divisible n_local and duplicate IDs."""
    devices = np.array(jax.devices())
    if len(devices) >= 4:
      mesh = Mesh(devices[:4].reshape(2, 2), ("fsdp", "ep"))
      row_spec = ("fsdp", "ep")
      n_local = 37  # Odd & non-divisible by num_chunks=5 (mirrors 320,001,920 / 128 = 2,500,015).
      vocab_size = n_local * 4
    else:
      mesh = Mesh(devices.reshape(-1, 1), ("fsdp", "ep"))
      row_spec = ("fsdp",)
      n_local = 37
      vocab_size = n_local * len(devices)
    width = 16

    class _ToyEmbedModel(nnx.Module):
      """Minimal embedding + linear test module."""

      def __init__(self, rngs: nnx.Rngs):
        self.ple_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=width,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            embedding_init=nnx.initializers.normal(0.1),
            rngs=rngs,
        )
        self.proj = nnx.Linear(width, width, use_bias=False, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

      def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.proj(self.ple_embedding(inputs))

    class _ToyCfg:
      parameter_memory_host_offload = False
      optimizer_memory_host_offload = False
      selective_host_offload = True
      host_offload_param_patterns = ("(?:^|/)ple_embedding/embedding$",)
      selective_host_offload_optimizer_mode = "host_compute"
      selective_host_offload_num_chunks = 5

    def _build(selective: bool, opt_mode: str = "host_compute"):
      model = _ToyEmbedModel(nnx.Rngs(0))
      sh_tbl = NamedSharding(mesh, P(row_spec, None))
      model.ple_embedding.embedding.set_metadata("out_sharding", P(row_spec, None))
      model.ple_embedding.embedding.set_raw_value(jax.device_put(model.ple_embedding.embedding.get_raw_value(), sh_tbl))
      model.proj.kernel.set_raw_value(
          jax.device_put(model.proj.kernel.get_raw_value(), NamedSharding(mesh, P(None, None)))
      )

      base_tx = optax.adamw(learning_rate=1e-2, weight_decay=0.1)
      if not selective:
        tx = optax.chain(optax.clip_by_global_norm(0.5), base_tx)
        return model, nnx.Optimizer(model, tx, wrt=nnx.Param)
      cfg = _ToyCfg()
      cfg.selective_host_offload_optimizer_mode = opt_mode
      model = m3_offload.maybe_offload_parameters(model, cfg)
      opt = m3_offload.make_optimizer(model, base_tx, cfg, clip_max_norm=0.5)
      return model, opt

    def _loss(m, batch):
      return jnp.mean(m(batch["inputs"]) ** 2)

    @nnx.jit(donate_argnums=(0, 1))
    def _step(m, opt, batch):
      if m3_offload.has_selective_host_offload(m):
        loss, grads = m3_offload.value_and_grad(_loss, m, batch)
      else:
        loss, grads = nnx.value_and_grad(_loss)(m, batch)
      opt.update(m, grads)
      return loss

    with jax.set_mesh(mesh):
      m_ref, opt_ref = _build(False)
      m_hc, opt_hc = _build(True, "host_compute")
      m_sc, opt_sc = _build(True, "streamed_chunks")
      m_ls, opt_ls = _build(True, "lazy_sparse")

      # Construct batches with explicit duplicate row IDs across shards and untouched rows.
      batches = []
      for s in range(3):
        raw_ids = jax.random.randint(jax.random.key(50 + s), (4, 8), 0, vocab_size, dtype=jnp.int32)
        # Force duplicate hits on row 3 and row n_local + 2 across shards.
        raw_ids = raw_ids.at[:, :2].set(jnp.array([3, min(n_local + 2, vocab_size - 1)], dtype=jnp.int32))
        batches.append({"inputs": jax.device_put(raw_ids, NamedSharding(mesh, P(None, None)))})

      # Step-0 exact gradient and global-norm parity (including duplicate-ID accumulation).
      l_ref0, g_ref0 = nnx.value_and_grad(_loss)(m_ref, batches[0])
      l_hc0, g_hc0 = m3_offload.value_and_grad(_loss, m_hc, batches[0])
      np.testing.assert_allclose(float(l_hc0), float(l_ref0), rtol=1e-6, atol=1e-6)
      norm_ref = float(m3_offload.l2norm_pytree(g_ref0))
      norm_hc = float(m3_offload.l2norm_pytree(g_hc0))
      np.testing.assert_allclose(norm_hc, norm_ref, rtol=1e-5, atol=1e-6)

      # Run 3 optimizer steps across all modes.
      losses_ref, losses_hc, losses_sc, losses_ls = [], [], [], []
      for b in batches:
        losses_ref.append(float(_step(m_ref, opt_ref, b)))
        losses_hc.append(float(_step(m_hc, opt_hc, b)))
        losses_sc.append(float(_step(m_sc, opt_sc, b)))
        losses_ls.append(float(_step(m_ls, opt_ls, b)))

      np.testing.assert_allclose(losses_hc, losses_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(losses_sc, losses_ref, rtol=1e-5, atol=1e-5)
      p_ref = _param_numpy_map(m_ref)
      p_hc = _param_numpy_map(m_hc)
      p_sc = _param_numpy_map(m_sc)
      p_ls = _param_numpy_map(m_ls)
      for path in p_ref:
        np.testing.assert_allclose(p_hc[path], p_ref[path], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(p_sc[path], p_ref[path], rtol=1e-5, atol=1e-5)

      # Verify opt-in lazy_sparse intentionally skips weight decay on untouched rows (documenting deviation).
      emb_path = ("ple_embedding", "embedding")
      touched = set(int(x) for b in batches for x in np.asarray(b["inputs"]).reshape(-1))
      untouched = [i for i in range(vocab_size) if i not in touched]
      self.assertNotEmpty(untouched)
      # Dense AdamW decayed untouched rows; lazy_sparse left them untouched, so they must diverge.
      untouched_diff = float(np.max(np.abs(p_ls[emb_path][untouched] - p_ref[emb_path][untouched])))
      self.assertGreater(untouched_diff, 1e-4)

      # Negative controls executing real m3_offload: bypassing step-1 AdamW bias correction (pre-advancing
      # ScaleByAdamState count so 1 - beta^t ~ 1) or perturbing SparseHostGrad.d_gathered before opt.update
      # breaks step-1 parity against the dense reference far above tolerance.
      m_ref1, opt_ref1 = _build(False)
      _step(m_ref1, opt_ref1, batches[0])
      p_ref1 = _param_numpy_map(m_ref1)

      m_nobc, opt_nobc = _build(True, "host_compute")
      found_count = False
      for leaf in jax.tree.leaves(opt_nobc.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
        if isinstance(leaf, nnx.Variable):
          val = leaf.get_raw_value()
          if jnp.issubdtype(val.dtype, jnp.integer) and val.ndim == 0:
            leaf.set_raw_value(jax.device_put(jnp.asarray(100_000, dtype=val.dtype), val.sharding))
            found_count = True
      self.assertTrue(found_count)
      _step(m_nobc, opt_nobc, batches[0])
      self.assertFalse(np.allclose(_param_numpy_map(m_nobc)[emb_path], p_ref1[emb_path], rtol=1e-5, atol=1e-5))

      m_pert, opt_pert = _build(True, "host_compute")
      _, g_pert = m3_offload.value_and_grad(_loss, m_pert, batches[0])
      sparse_hits = 0

      def _perturb_sparse(x):
        nonlocal sparse_hits
        if isinstance(x, m3_offload.SparseHostGrad):
          sparse_hits += 1
          return x.with_d_gathered(x.d_gathered + jnp.asarray(0.1, dtype=x.d_gathered.dtype))
        return x

      g_pert = jax.tree.map(_perturb_sparse, g_pert, is_leaf=lambda x: isinstance(x, m3_offload.SparseHostGrad))
      self.assertGreater(sparse_hits, 0)
      opt_pert.update(m_pert, g_pert)
      self.assertFalse(np.allclose(_param_numpy_map(m_pert)[emb_path], p_ref1[emb_path], rtol=1e-5, atol=1e-5))

    # Verify row-sharded `streamed_chunks` also isolates inside `jax.shard_map` when called OUTSIDE
    # `with jax.set_mesh(mesh):` (emitting 0 MegaScale collectives and matching captured param sqnorm).
    if len(devices) >= 2:
      with jax.set_mesh(mesh):
        m_ref_no_ctx, opt_ref_no_ctx = _build(False)
        m_sc_no_ctx, opt_sc_no_ctx = _build(True, "streamed_chunks")
        m_ls_no_ctx, opt_ls_no_ctx = _build(True, "lazy_sparse")

      @nnx.jit(donate_argnums=(0, 1))
      def _step_capture(m, opt, batch):
        if m3_offload.has_selective_host_offload(m):
          loss, grads = m3_offload.value_and_grad(_loss, m, batch)
          with m3_offload.capture_host_param_sqnorms() as sq_map:
            opt.update(m, grads)
          assert sq_map, "capture_host_param_sqnorms() did not populate sq_map"
          pnorm = m3_offload.l2norm_pytree_with_overrides(nnx.state(m, nnx.Param), sq_map)
        else:
          loss, grads = nnx.value_and_grad(_loss)(m, batch)
          opt.update(m, grads)
          pnorm = m3_offload.l2norm_pytree(nnx.state(m, nnx.Param))
        return loss, pnorm

      self.assertTrue(jax.sharding.get_abstract_mesh().empty)
      compiled_no_ctx = _step_capture.lower(m_sc_no_ctx, opt_sc_no_ctx, batches[0]).compile()
      self.assertNotIn("MegaScaleCollective", compiled_no_ctx.as_text())
      compiled_ls_no_ctx = _step_capture.lower(m_ls_no_ctx, opt_ls_no_ctx, batches[0]).compile()
      self.assertNotIn("MegaScaleCollective", compiled_ls_no_ctx.as_text())
      _, pn_ls0 = _step_capture(m_ls_no_ctx, opt_ls_no_ctx, batches[0])
      with jax.set_mesh(mesh):
        expected_pn_ls0 = float(m3_offload.l2norm_pytree(nnx.state(m_ls_no_ctx, nnx.Param)))
      np.testing.assert_allclose(float(pn_ls0), expected_pn_ls0, rtol=1e-5, atol=1e-5)
      for b in batches:
        l_r, pn_r = _step_capture(m_ref_no_ctx, opt_ref_no_ctx, b)
        l_s, pn_s = _step_capture(m_sc_no_ctx, opt_sc_no_ctx, b)
        np.testing.assert_allclose(float(l_s), float(l_r), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(pn_s), float(pn_r), rtol=1e-5, atol=1e-5)
      p_r_map = _param_numpy_map(m_ref_no_ctx)
      p_s_map = _param_numpy_map(m_sc_no_ctx)
      for path in p_r_map:
        np.testing.assert_allclose(p_s_map[path], p_r_map[path], rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("scanned", True),
      ("unscanned", False),
  )
  def test_qwen38_mini_eval_shape_and_step0_grad_parity(self, scan_layers: bool):
    """Verifies `eval_shape` `pinned_host` placement and step-0 loss/gradient parity."""
    out_dir = self.create_tempdir().full_path
    cfg_off = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="false",
    )
    cfg_on = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="true",
    )
    mesh = mesh_from_config(cfg_off)
    sharding = sharding_from_config(cfg_off)

    with jax.set_mesh(mesh):
      abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg_on, mesh, sharding))
      abs_ple_paths = m3_offload.selected_host_param_paths(abs_model)
      self.assertEqual(len(abs_ple_paths), 1)
      for path, var in nnx.to_flat_state(nnx.state(abs_model, nnx.Param)):
        mk = var.get_raw_value().sharding.memory_kind
        if tuple(path) in abs_ple_paths:
          self.assertEqual(mk, "pinned_host")
        else:
          self.assertIn(mk, (None, "device"))
      abs_host_opt = 0
      abs_dev_opt = 0
      for path, leaf in jax.tree_util.tree_leaves_with_path(
          abs_opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      ):
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          continue
        key_t = tuple(m3_offload._key_value(k) for k in path)  # pylint: disable=protected-access
        is_sel = m3_offload._is_matching_opt_leaf(key_t, abs_ple_paths) is not None  # pylint: disable=protected-access
        if is_sel:
          self.assertEqual(raw.sharding.memory_kind, "pinned_host")
        else:
          self.assertIn(raw.sharding.memory_kind, (None, "device"))
        abs_host_opt += int(is_sel)
        abs_dev_opt += int(not is_sel)
      self.assertEqual(abs_host_opt, 2)
      self.assertGreater(abs_dev_opt, 10)

      m_off, _ = loop.setup_model(cfg_off, mesh, sharding)
      m_on, _ = loop.setup_model(cfg_on, mesh, sharding)
      self.assertEqual(_param_paths(m_on), _param_paths(m_off))

      batch0 = loop.make_synthetic_batch(cfg_off, mesh, sharding, jax.random.key(0))
      l0_off, g0_off = nnx.jit(nnx.value_and_grad(loop.loss_fn))(m_off, batch0)
      l0_on, g0_on = nnx.jit(lambda m, b: m3_offload.value_and_grad(loop.loss_fn, m, b))(m_on, batch0)
      np.testing.assert_allclose(float(l0_on), float(l0_off), rtol=1e-6, atol=1e-6)
      g_off_map = {tuple(p): v.get_raw_value() for p, v in nnx.to_flat_state(g0_off)}
      g_on_map = {
          tuple(p): (v.get_raw_value() if isinstance(v, nnx.Variable) else v) for p, v in nnx.to_flat_state(g0_on)
      }
      for path, go in g_off_map.items():
        gn = g_on_map[path]
        if isinstance(gn, m3_offload.SparseHostGrad):
          gn = m3_offload.materialize_sparse_grad(gn)
        np.testing.assert_allclose(np.asarray(gn), np.asarray(go), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("scanned", True),
      ("unscanned", False),
  )
  def test_qwen38_mini_loop_parity_and_memory_kind(self, scan_layers: bool):
    """Verifies 3-step `loop.train_step` trajectory parity, post-step `pinned_host` invariants, and PLE perturbation."""
    out_dir = self.create_tempdir().full_path
    cfg_off = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="false",
    )
    cfg_on = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="true",
    )
    mesh = mesh_from_config(cfg_off)
    sharding = sharding_from_config(cfg_off)

    with jax.set_mesh(mesh):
      m_off, opt_off = loop.setup_model(cfg_off, mesh, sharding)
      m_on, opt_on = loop.setup_model(cfg_on, mesh, sharding)
      self.assertEqual(_param_paths(m_on), _param_paths(m_off))
      batch0 = loop.make_synthetic_batch(cfg_off, mesh, sharding, jax.random.key(0))

      # 3-step exact AdamW trajectory parity (step-0 elementwise <= 1e-6; 3-step Frobenius <= 1e-5).
      losses_off, losses_on = [], []
      for step in range(3):
        batch = loop.make_synthetic_batch(cfg_off, mesh, sharding, jax.random.key(10 + step))
        losses_off.append(float(loop.train_step(m_off, opt_off, batch)))
        losses_on.append(float(loop.train_step(m_on, opt_on, batch)))
        if step == 0:
          p_off0 = _param_numpy_map(m_off)
          p_on0 = _param_numpy_map(m_on)
          for path in p_off0:
            np.testing.assert_allclose(p_on0[path], p_off0[path], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(losses_on, losses_off, rtol=1e-5, atol=1e-5)
      p_off = _param_numpy_map(m_off)
      p_on = _param_numpy_map(m_on)
      for path in p_off:
        # Damp denominator at 1.0 for zero-centered `1+scale` norms (`||scale||_2 ~ 0.01`, max_d < 1e-6).
        rel_frob = float(np.linalg.norm(p_on[path] - p_off[path]) / max(float(np.linalg.norm(p_off[path])), 1.0))
        self.assertLess(rel_frob, 1e-5, f"Frobenius drift too large on {path}: {rel_frob:.2e}")
        np.testing.assert_allclose(p_on[path], p_off[path], rtol=1e-4, atol=1e-4)

      # Post-step memory_kind invariants: ONLY PLE weight + its mu/nu remain in pinned_host.
      ple_paths = m3_offload.selected_host_param_paths(m_on)
      for path, var in nnx.to_flat_state(nnx.state(m_on, nnx.Param)):
        mk = var.get_raw_value().sharding.memory_kind
        if tuple(path) in ple_paths:
          self.assertEqual(mk, "pinned_host")
        else:
          self.assertIn(mk, (None, "device"))
      post_host_opt = 0
      for path, leaf in jax.tree_util.tree_leaves_with_path(
          opt_on.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      ):
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          continue
        key_t = tuple(m3_offload._key_value(k) for k in path)  # pylint: disable=protected-access
        is_sel = m3_offload._is_matching_opt_leaf(key_t, ple_paths) is not None  # pylint: disable=protected-access
        if is_sel:
          self.assertEqual(raw.sharding.memory_kind, "pinned_host")
          post_host_opt += 1
        else:
          self.assertIn(raw.sharding.memory_kind, (None, "device"))
      self.assertEqual(post_host_opt, 2)

      # Negative control: perturbing the offloaded PLE table must break loss parity.
      ple_path = next(iter(ple_paths))
      ple_var = dict(nnx.to_flat_state(nnx.state(m_on, nnx.Param)))[ple_path]
      ple_var.set_raw_value(m3_offload.to_memory(ple_var.get_raw_value() + 1.0, "pinned_host"))
      jit_loss = nnx.jit(loop.loss_fn)
      perturbed_loss = float(jit_loss(m_on, batch0))
      ref_loss = float(jit_loss(m_off, batch0))
      self.assertGreater(abs(perturbed_loss - ref_loss), 1e-3)

  def test_default_off_hlo_byte_identical(self):
    """Verifies Qwen3.8 default-off (`selective_host_offload=False`) emits byte-identical HLO."""
    out_dir = self.create_tempdir().full_path
    cfg_default = _qwen38_mini_config(out_dir, max_target_length="8", scan_layers="true", selective_host_offload="false")
    cfg_empty_pat = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers="true",
        selective_host_offload="false",
        host_offload_param_patterns="[]",
    )
    self.assertNotEmpty(cfg_default.host_offload_param_patterns)
    mesh = mesh_from_config(cfg_default)
    sharding = sharding_from_config(cfg_default)

    cfg_selective_on = _qwen38_mini_config(
        out_dir, max_target_length="8", scan_layers="true", selective_host_offload="true"
    )

    def _compile_hlo(cfg):
      with jax.set_mesh(mesh):
        m, opt = loop.setup_model(cfg, mesh, sharding)
        batch = loop.make_synthetic_batch(cfg, mesh, sharding, jax.random.key(0))
        return _normalize_hlo(loop.train_step.lower(m, opt, batch).compile().as_text())

    hlo_default = _compile_hlo(cfg_default)
    self.assertEqual(hlo_default, _compile_hlo(cfg_empty_pat))
    self.assertNotEqual(hlo_default, _compile_hlo(cfg_selective_on))

  def test_dual_drivers_and_transform_composition(self):
    """Verifies selective offload loss, grad-norm, param-norm, and memory kinds in `train.py`."""
    out_dir = self.create_tempdir().full_path

    def _run_legacy(selective: bool):
      cfg = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          dataset_type="synthetic",
          selective_host_offload=str(selective).lower(),
      )
      mesh = mesh_from_config(cfg)
      with jax.set_mesh(mesh):
        (
            _,
            _,
            state_mesh_shardings,
            _,
            mesh,
            _,
            data_iter,
            _,
            _,
            _,
            train_state,
        ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
        jit_model, state = nnx.split(train_state)
        p_step, _ = train_utils.jit_train_and_eval_step(
            cfg, jit_model, mesh, state, state_mesh_shardings, legacy_train.train_step
        )
        losses, gnorms, pnorms = [], [], []
        for _ in range(2):
          batch = next(data_iter)
          state, metrics = p_step(state, batch)
          losses.append(float(metrics["scalar"]["learning/loss"]))
          gnorms.append(float(metrics["scalar"]["learning/grad_norm"]))
          pnorms.append(float(metrics["scalar"]["learning/param_norm"]))
      return state, losses, gnorms, pnorms

    _, l_off, gn_off, pn_off = _run_legacy(False)
    st_on, l_on, gn_on, pn_on = _run_legacy(True)
    np.testing.assert_allclose(l_on, l_off, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gn_on, gn_off, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pn_on, pn_off, rtol=1e-4, atol=1e-4)
    host_m = [
        x for x in jax.tree.leaves(st_on.model) if hasattr(x, "sharding") and x.sharding.memory_kind == "pinned_host"
    ]
    dev_m = [x for x in jax.tree.leaves(st_on.model) if hasattr(x, "sharding") and x.sharding.memory_kind == "device"]
    host_o = [
        x
        for x in jax.tree.leaves(st_on.optimizer)
        if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0 and x.sharding.memory_kind == "pinned_host"
    ]
    dev_o = [
        x
        for x in jax.tree.leaves(st_on.optimizer)
        if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0 and x.sharding.memory_kind == "device"
    ]
    self.assertLen(host_m, 1)
    self.assertGreater(len(dev_m), 10)
    self.assertLen(host_o, 2)
    self.assertGreater(len(dev_o), 10)

  def test_transform_composition_scan_remat_qwix_and_selective_offload(self):
    """Verifies `scan` + `remat=full` + Qwix `int8` quantization + selective host offload path equality."""
    out_dir = self.create_tempdir().full_path
    cfg_comp = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        remat_policy="full",
        quantization="int8",
        use_qwix_quantization="true",
        selective_host_offload="true",
    )
    cfg_comp_off = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        remat_policy="full",
        quantization="int8",
        use_qwix_quantization="true",
        selective_host_offload="false",
    )
    mesh = mesh_from_config(cfg_comp)
    with jax.set_mesh(mesh):
      m_comp_off = model_creation_utils.create_model(cfg_comp_off, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_comp_on = model_creation_utils.create_model(cfg_comp, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertEqual(_param_paths(m_comp_on), _param_paths(m_comp_off))
      ple_paths = m3_offload.selected_host_param_paths(m_comp_on)
      self.assertLen(ple_paths, 1)
      toks = jnp.ones((cfg_comp.global_batch_size_to_train_on, 16), dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), toks.shape)
      self.assertGreater(m3_quant.quantized_matmuls(m_comp_on, jnp.int8, toks, pos), 0)

  def test_dual_drivers_maxtext_training_engine_parity(self):
    """Verifies selective offload 2-step loss parity in `MaxTextTrainingEngine`."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    mesh = mesh_from_config(_qwen38_mini_config(out_dir, max_target_length="8", mtp_num_layers="0", scan_layers="true"))

    def _run_engine(selective: bool):
      cfg_e = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          selective_host_offload=str(selective).lower(),
      )
      eng = maxtext_engine.MaxTextTrainingEngine(cfg_e, mesh=mesh)
      bs = cfg_e.global_batch_size_to_train_on
      seq = cfg_e.max_target_length
      losses = []
      for step in range(2):
        toks = jax.random.randint(jax.random.key(200 + step), (bs, seq), 0, cfg_e.vocab_size, dtype=jnp.int32)
        pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (bs, seq))
        seg = jnp.ones((bs, seq), dtype=jnp.int32)
        batch = {
            "inputs": toks,
            "targets": jnp.roll(toks, -1, axis=-1),
            "inputs_position": pos,
            "targets_position": pos,
            "inputs_segmentation": seg,
            "targets_segmentation": seg,
        }
        if step == 0:
          eng.compile(batch)
        eng.fwd_bwd(batch)
        eng.update()
        m = eng.get_metrics()
        lk = next(iter(m.weighted_metrics))
        losses.append(float(jnp.sum(m.weighted_metrics[lk].unreduced_sum) / jnp.sum(m.weighted_metrics[lk].denominator)))
      jax.block_until_ready(eng._read_state_pure())  # pylint: disable=protected-access
      return losses

    np.testing.assert_allclose(_run_engine(True), _run_engine(False), rtol=1e-5, atol=1e-5)

  def _compile_tpu7x_selective_ple(
      self, out_dir: str, topo, *, selective: bool, opt_mode: str = "host_compute", row_shard_ple: bool = False
  ):
    """Compiles Qwen3.8 mini with a 100K PLE table on virtual `tpu7x-8` and returns memory analysis and HLO."""
    cfg = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        compile_topology="tpu7x-8",
        ici_fsdp_parallelism="4",
        ici_expert_parallelism="2",
        scan_layers="true",
        remat_policy="full",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        ngram_vocab_size_base="100000",
        selective_host_offload=str(selective).lower(),
        selective_host_offload_optimizer_mode=opt_mode,
        selective_host_offload_num_chunks="8",
    )
    tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
    sharding = sharding_from_config(cfg)
    with jax.set_mesh(tpu_mesh):
      data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
      data_sharding = NamedSharding(tpu_mesh, data_spec)
      batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
      batch = {
          "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
          "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
          "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
      }
      abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh))
      if row_shard_ple:
        ple_paths = m3_offload._match_param_paths(  # pylint: disable=protected-access
            nnx.state(abs_model, nnx.Param), cfg.host_offload_param_patterns
        )
        self.assertLen(ple_paths, 1)
        emb_var = dict(nnx.to_flat_state(nnx.state(abs_model, nnx.Param)))[next(iter(ple_paths))]
        emb_sds = emb_var.get_raw_value()
        emb_kind = emb_sds.sharding.memory_kind
        emb_var.set_metadata("out_sharding", P("fsdp", None))
        emb_var.set_raw_value(
            jax.ShapeDtypeStruct(
                emb_sds.shape,
                emb_sds.dtype,
                sharding=NamedSharding(tpu_mesh, P("fsdp", None), memory_kind=emb_kind),
            )
        )
        for leaf in jax.tree.leaves(abs_opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
          if isinstance(leaf, nnx.Variable):
            raw = leaf.get_raw_value()
            if hasattr(raw, "shape") and raw.shape == emb_sds.shape:
              leaf.set_raw_value(
                  jax.ShapeDtypeStruct(
                      raw.shape,
                      raw.dtype,
                      sharding=NamedSharding(tpu_mesh, P("fsdp", None), memory_kind=raw.sharding.memory_kind),
                  )
              )
      compiled = loop.train_step.lower(abs_model, abs_opt, batch).compile()
    rows = memory_report.memory_report({"update": compiled}, {})
    return compiled.memory_analysis(), rows[0], compiled.as_text() or ""

  @pytest.mark.tpu_backend
  @parameterized.named_parameters(
      ("host_compute", "host_compute"),
      ("streamed_chunks", "streamed_chunks"),
  )
  def test_tpu7x_aot_selective_ple_offload_reduces_hbm_without_dense_grad(self, opt_mode: str):
    """Compiles unsharded Qwen3.8 mini PLE offload on virtual `tpu7x-8` and checks HBM drop."""
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path
    topo = _get_tpu7x8_topology()

    cache = type(self)._tpu7x_ple_unsharded_off_cache  # pylint: disable=protected-access
    if "off" not in cache:
      mem_off, row_off, _ = self._compile_tpu7x_selective_ple(out_dir, topo, selective=False)
      cache["off"] = (int(mem_off.host_argument_size_in_bytes), int(row_off.argument), int(row_off.device_total))
    off_host_arg_bytes, off_arg, off_dev_total = cache["off"]

    _, row_sel, hlo_sel = self._compile_tpu7x_selective_ple(
        out_dir, topo, selective=True, opt_mode=opt_mode, row_shard_ple=False
    )
    self.assertEqual(off_host_arg_bytes, 0)
    self.assertNotIn("MegaScaleCollective", hlo_sel)
    arg_drop = off_arg - row_sel.argument
    self.assertGreater(arg_drop, 140 * 1024 * 1024)
    self.assertEqual(row_sel.host_argument, arg_drop)
    self.assertEqual(row_sel.host_alias, row_sel.host_argument)
    self.assertLess(row_sel.device_total, off_dev_total - 100 * 1024 * 1024)
    if opt_mode == "host_compute":
      self.assertLessEqual(row_sel.host_temp, row_sel.host_argument)
    else:
      jax_ver = tuple(int(part) for part in jax.__version__.split(".")[:2] if part.isdigit())
      if jax_ver >= (0, 12):
        self.assertEqual(row_sel.host_temp, 0)
      else:
        self.assertLessEqual(row_sel.host_temp, 2 * row_sel.host_argument + 1024 * 1024)

  @pytest.mark.tpu_backend
  def test_tpu7x_aot_row_sharded_selective_ple_streamed_chunks(self):
    """Verifies row-sharded (`P('fsdp', None)`) `streamed_chunks` emits 0 MegaScale ops and cuts HBM."""
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path
    topo = _get_tpu7x8_topology()

    _, row_off_sharded, _ = self._compile_tpu7x_selective_ple(out_dir, topo, selective=False, row_shard_ple=True)
    _, row_sc_sharded, hlo_sc_sharded = self._compile_tpu7x_selective_ple(
        out_dir, topo, selective=True, opt_mode="streamed_chunks", row_shard_ple=True
    )

    self.assertNotIn("MegaScaleCollective", hlo_sc_sharded)
    sharded_arg_drop = row_off_sharded.argument - row_sc_sharded.argument
    self.assertGreater(sharded_arg_drop, 35 * 1024 * 1024)
    self.assertEqual(row_sc_sharded.host_argument, sharded_arg_drop)
    self.assertEqual(row_sc_sharded.host_alias, row_sc_sharded.host_argument)
    self.assertLess(row_sc_sharded.device_total, row_off_sharded.device_total)
    jax_ver = tuple(int(part) for part in jax.__version__.split(".")[:2] if part.isdigit())
    if jax_ver >= (0, 12):
      self.assertEqual(row_sc_sharded.host_temp, 0)
      self.assertLess(row_sc_sharded.device_total, row_off_sharded.device_total - 80 * 1024 * 1024)
    else:
      self.assertLessEqual(row_sc_sharded.host_temp, 2 * row_sc_sharded.host_argument + 1024 * 1024)
      self.assertLess(row_sc_sharded.host_temp, 90 * 1024 * 1024)

  @pytest.mark.tpu_backend
  @parameterized.named_parameters(
      ("host_compute", "host_compute"),
      ("streamed_chunks", "streamed_chunks"),
  )
  def test_tpu7x_aot_engine_selective_ple_offload_reduces_hbm_zero_megascale(self, opt_mode: str):
    """Verifies `AbstractMaxTextEngine` AOT PLE offload emits 0 MegaScale collectives and cuts device peak."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path

    def _compile_engine(selective: bool, mode: str = "host_compute"):
      cfg = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          compile_topology="tpu7x-8",
          compile_topology_num_slices="1",
          ici_fsdp_parallelism="4",
          ici_expert_parallelism="2",
          scan_layers="true",
          remat_policy="full",
          dtype="bfloat16",
          weight_dtype="bfloat16",
          ngram_vocab_size_base="100000",
          selective_host_offload=str(selective).lower(),
          selective_host_offload_optimizer_mode=mode,
          selective_host_offload_num_chunks="8",
      )
      tpu_mesh = train_compile.get_topology_mesh(cfg)
      eng = maxtext_engine_compile.AbstractMaxTextEngine(cfg, tpu_mesh)
      compiled = eng.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
      rows = memory_report.memory_report(compiled, eng.train_state_avals())
      peak = memory_report.device_peak(rows)
      ms_count = sum((c.as_text() or "").count("MegaScaleCollective") for c in compiled.values())
      return peak, ms_count

    cache = type(self)._tpu7x_engine_ple_off_cache  # pylint: disable=protected-access
    if "off" not in cache:
      peak_off, ms_off = _compile_engine(False)
      cache["off"] = (int(peak_off.device_total), int(ms_off))
    off_dev_total, ms_off = cache["off"]

    peak_sel, ms_sel = _compile_engine(True, opt_mode)
    self.assertEqual(ms_off, 0)
    self.assertEqual(ms_sel, 0)
    self.assertLess(peak_sel.device_total, off_dev_total - 100 * 1024 * 1024)

  def test_explicit_sharding_and_engine_zero1(self):
    """Verifies tied `qwen3-0.6b` under `shard_mode='explicit'` in `loop.py` and fail-closed `lazy_sparse`."""
    out_dir = self.create_tempdir().full_path
    dp = str(len(jax.devices()))

    cfg_tied_ref = _tiny_m3_config(
        out_dir,
        per_device_batch_size="1",
        max_target_length="8",
        shard_mode="explicit",
        ici_data_parallelism=dp,
        ici_fsdp_parallelism="1",
        ici_tensor_parallelism="1",
        selective_host_offload="false",
    )
    mesh = mesh_from_config(cfg_tied_ref)
    sharding = sharding_from_config(cfg_tied_ref)
    with jax.set_mesh(mesh):
      m_ref, opt_ref = loop.setup_model(cfg_tied_ref, mesh, sharding)
      batch_tied = loop.make_synthetic_batch(cfg_tied_ref, mesh, sharding, jax.random.key(0))
      loss_tied_ref = float(loop.train_step(m_ref, opt_ref, batch_tied))

    for mode in ("host_compute", "streamed_chunks"):
      cfg_tied = _tiny_m3_config(
          out_dir,
          per_device_batch_size="1",
          max_target_length="8",
          shard_mode="explicit",
          ici_data_parallelism=dp,
          ici_fsdp_parallelism="1",
          ici_tensor_parallelism="1",
          selective_host_offload="true",
          host_offload_param_patterns="['^token_embedder/embedding$']",
          selective_host_offload_optimizer_mode=mode,
      )
      with jax.set_mesh(mesh):
        m_t, opt_t = loop.setup_model(cfg_tied, mesh, sharding)
        loss_t = float(loop.train_step(m_t, opt_t, batch_tied))
      np.testing.assert_allclose(loss_t, loss_tied_ref, rtol=1e-5, atol=1e-5, err_msg=f"tied explicit {mode}")

    cfg_tied_ls = _tiny_m3_config(
        out_dir,
        per_device_batch_size="1",
        max_target_length="8",
        shard_mode="explicit",
        ici_data_parallelism=dp,
        ici_fsdp_parallelism="1",
        ici_tensor_parallelism="1",
        selective_host_offload="true",
        host_offload_param_patterns="['^token_embedder/embedding$']",
        selective_host_offload_optimizer_mode="lazy_sparse",
    )
    with self.assertRaisesRegex(ValueError, "SparseHostGrad"):
      with jax.set_mesh(mesh):
        m_ls, opt_ls = loop.setup_model(cfg_tied_ls, mesh, sharding)
        loop.train_step(m_ls, opt_ls, batch_tied)

  def test_explicit_sharding_untied_size1_axes_all_modes_parity(self):
    """Verifies untied `HostOffloadedEmbed` on a size-1-axis `AxisType.Explicit` mesh across all 3 modes."""
    devs = np.array(jax.devices())
    exp_mesh = Mesh(
        devs.reshape((len(devs), 1, 1)),
        ("data", "fsdp", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 3,
    )

    class _UntiedExpModel(nnx.Module):
      """Minimal untied embedding + linear head with explicit size-1 axis metadata."""

      def __init__(self, rngs: nnx.Rngs):
        self.emb = nnx.Embed(
            num_embeddings=64,
            features=16,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            embedding_metadata={"out_sharding": P("tensor", "fsdp")},
            rngs=rngs,
        )
        self.w = nnx.Param(
            jax.random.normal(rngs.params(), (16, 32), dtype=jnp.float32),
            out_sharding=P("fsdp", "tensor"),
        )

      def __call__(self, toks: jax.Array) -> jax.Array:
        if isinstance(self.emb, m3_offload.HostOffloadedEmbed):
          x = self.emb(toks)
        else:
          x = self.emb.embedding[...].at[toks].get(out_sharding=P("data", None, "fsdp"))
        return jnp.einsum("bsd,do->bso", x, self.w[...], out_sharding=P("data", None, "tensor"))

    class _UntiedCfg:
      """Minimal selective offload config for `_UntiedExpModel`."""

      parameter_memory_host_offload = False
      optimizer_memory_host_offload = False
      selective_host_offload = True
      host_offload_param_patterns = ("^emb/embedding$",)
      selective_host_offload_optimizer_mode = "host_compute"
      selective_host_offload_num_chunks = 4

    def _u_loss(mod, b):
      return jnp.mean(mod(b["inputs"]) ** 2)

    with jax.set_mesh(exp_mesh):
      b_exp = {
          "inputs": jax.device_put(
              jax.random.randint(jax.random.key(5), (len(devs), 8), 0, 64, dtype=jnp.int32),
              NamedSharding(exp_mesh, P("data", None)),
          )
      }
      m_uref = _UntiedExpModel(nnx.Rngs(0))
      opt_uref = nnx.Optimizer(m_uref, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2)), wrt=nnx.Param)

      @nnx.jit
      def _u_ref_step(mod, opt_inst, batch_in):
        loss_val, grads_val = nnx.value_and_grad(_u_loss)(mod, batch_in)
        opt_inst.update(mod, grads_val)
        return loss_val

      loss_uref0 = float(_u_ref_step(m_uref, opt_uref, b_exp))
      loss_uref1 = float(_u_ref_step(m_uref, opt_uref, b_exp))

      for mode in ("host_compute", "streamed_chunks", "lazy_sparse"):
        ucfg = _UntiedCfg()
        ucfg.selective_host_offload_optimizer_mode = mode
        m_u = m3_offload.maybe_offload_parameters(_UntiedExpModel(nnx.Rngs(0)), ucfg)
        opt_u = m3_offload.make_optimizer(m_u, optax.adamw(1e-2), ucfg, clip_max_norm=1.0)

        @nnx.jit
        def _u_step(mod, opt_inst, batch_in):
          loss_val, grads_val = m3_offload.value_and_grad(_u_loss, mod, batch_in)
          opt_inst.update(mod, grads_val)
          return loss_val

        loss_u0 = float(_u_step(m_u, opt_u, b_exp))
        loss_u1 = float(_u_step(m_u, opt_u, b_exp))
        np.testing.assert_allclose(loss_u0, loss_uref0, rtol=1e-5, atol=1e-5, err_msg=f"untied explicit s0 {mode}")
        if mode != "lazy_sparse":
          np.testing.assert_allclose(loss_u1, loss_uref1, rtol=1e-5, atol=1e-5, err_msg=f"untied explicit s1 {mode}")

  @parameterized.named_parameters(("no_skip_nan", False), ("skip_nan", True))
  def test_engine_explicit_sharding_zero1_selective_offload(self, skip_nan: bool):
    """Verifies `MaxTextTrainingEngine` `shard_mode='explicit'` + ZeRO-1 loss parity under selective offload."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    dp = str(len(jax.devices()))
    cfg_base = _tiny_m3_config(
        out_dir,
        per_device_batch_size="1",
        max_target_length="8",
        shard_mode="explicit",
        ici_data_parallelism=dp,
        ici_fsdp_parallelism="1",
        ici_tensor_parallelism="1",
        selective_host_offload="false",
    )
    mesh = mesh_from_config(cfg_base)

    def _run_zero1_engine(selective: bool):
      cfg_e = _tiny_m3_config(
          out_dir,
          per_device_batch_size="1",
          max_target_length="8",
          shard_mode="explicit",
          ici_data_parallelism=dp,
          ici_fsdp_parallelism="1",
          ici_tensor_parallelism="1",
          shard_optimizer_over_data="true",
          skip_step_on_nan=str(skip_nan).lower(),
          selective_host_offload=str(selective).lower(),
          host_offload_param_patterns="['^token_embedder/embedding$']" if selective else "[]",
      )
      eng = maxtext_engine.MaxTextTrainingEngine(cfg_e, mesh=mesh)
      bs = cfg_e.global_batch_size_to_train_on
      seq = cfg_e.max_target_length
      data_sh = NamedSharding(mesh, P("data", None))
      losses = []
      for step in range(2):
        toks = jax.device_put(
            jax.random.randint(jax.random.key(10 + step), (bs, seq), 0, cfg_e.vocab_size, dtype=jnp.int32),
            data_sh,
        )
        pos = jax.device_put(jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (bs, seq)), data_sh)
        seg = jax.device_put(jnp.ones((bs, seq), dtype=jnp.int32), data_sh)
        batch = {
            "inputs": toks,
            "targets": jnp.roll(toks, -1, axis=-1),
            "inputs_position": pos,
            "targets_position": pos,
            "inputs_segmentation": seg,
            "targets_segmentation": seg,
        }
        if step == 0:
          eng.compile(batch)
        eng.fwd_bwd(batch)
        eng.update()
        m_out = eng.get_metrics()
        lk = next(iter(m_out.weighted_metrics))
        losses.append(
            float(jnp.sum(m_out.weighted_metrics[lk].unreduced_sum) / jnp.sum(m_out.weighted_metrics[lk].denominator))
        )
      return losses

    np.testing.assert_allclose(_run_zero1_engine(True), _run_zero1_engine(False), rtol=1e-5, atol=1e-5)

  def test_engine_eager_uncompiled_selective_offload_skips_device_staging(self):
    """Verifies eager (uncompiled) `MaxTextTrainingEngine.update()` never stages selective host optimizer state."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    cfg_eager = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        selective_host_offload="true",
        optimizer_memory_host_offload="true",
    )
    mesh_eager = mesh_from_config(cfg_eager)
    eng_eager = maxtext_engine.MaxTextTrainingEngine(cfg_eager, mesh=mesh_eager)
    bs_e = cfg_eager.global_batch_size_to_train_on
    toks_e = jax.random.randint(jax.random.key(42), (bs_e, 8), 0, cfg_eager.vocab_size, dtype=jnp.int32)
    pos_e = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (bs_e, 8))
    seg_e = jnp.ones((bs_e, 8), dtype=jnp.int32)
    batch_e = {
        "inputs": toks_e,
        "targets": jnp.roll(toks_e, -1, axis=-1),
        "inputs_position": pos_e,
        "targets_position": pos_e,
        "inputs_segmentation": seg_e,
        "targets_segmentation": seg_e,
    }
    with mock.patch.object(
        eng_eager,
        "_with_optimizer_on_device",
        wraps=eng_eager._with_optimizer_on_device,  # pylint: disable=protected-access
    ) as spy_on_dev:
      eng_eager.fwd_bwd(batch_e)
      eng_eager.update()
      spy_on_dev.assert_not_called()

  @pytest.mark.tpu_backend
  def test_tpu7x_engine_zero1_selective_offload_emits_zero_megascale_collectives(self):
    """Verifies AbstractMaxTextEngine ZeRO-1 + selective_host_offload on tpu7x-8 emits zero MegaScale ops."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path
    cfg = _tiny_m3_config(
        out_dir,
        model_name="qwen3-0.6b",
        compile_topology="tpu7x-8",
        compile_topology_num_slices="1",
        shard_mode="explicit",
        ici_data_parallelism="4",
        ici_fsdp_parallelism="1",
        ici_tensor_parallelism="2",
        shard_optimizer_over_data="true",
        selective_host_offload="true",
        host_offload_param_patterns="['^token_embedder/embedding$']",
        per_device_batch_size="1",
        max_target_length="32",
    )
    tpu_mesh = train_compile.get_topology_mesh(cfg)
    eng = maxtext_engine_compile.AbstractMaxTextEngine(cfg, tpu_mesh)
    compiled = eng.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
    self.assertNotEmpty(compiled)
    for step_name, comp in compiled.items():
      self.assertEqual((comp.as_text() or "").count("MegaScaleCollective"), 0, f"MegaScaleCollective in {step_name}")

    def _uses_data(spec: P) -> bool:
      return any(ax == "data" or (isinstance(ax, tuple) and "data" in ax) for ax in spec)

    opt_avals = [
        a for a in jax.tree.leaves(eng.train_state_avals()["optimizer"]) if getattr(a, "sharding", None) is not None
    ]
    self.assertTrue(
        any(getattr(a.sharding, "memory_kind", None) != "pinned_host" and _uses_data(a.sharding.spec) for a in opt_avals)
    )
    self.assertTrue(
        any(
            getattr(a.sharding, "memory_kind", None) == "pinned_host" and not _uses_data(a.sharding.spec)
            for a in opt_avals
        )
    )

  def test_skip_step_on_spikes_and_non2d_fail_closed(self):
    """Verifies non-2D fail-closed validation, forced NaN spike skipping, and duplicate-ID `step_diagnostics`."""
    from maxtext.optimizers import optimizers as mt_optimizers  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    cfg_1d = _tiny_m3_config(
        out_dir,
        selective_host_offload="true",
        host_offload_param_patterns="['.*norm/scale$']",
    )
    mesh_1d = mesh_from_config(cfg_1d)
    with jax.set_mesh(mesh_1d):
      with self.assertRaisesRegex(ValueError, "2D"):
        model_creation_utils.create_model(cfg_1d, mesh_1d, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    # Reject moe_dropless_fallback='step' with selective_host_offload=True, and reject non-nnx.Embed 2D matches.
    with self.assertRaisesRegex(ValueError, "selective_host_offload"):
      _qwen38_mini_config(
          out_dir,
          ici_fsdp_parallelism="1",
          ici_tensor_parallelism="1",
          ici_expert_parallelism=str(max(2, len(jax.devices()))),
          ragged_buffer_factor="1.5",
          use_ring_of_experts="true",
          use_ragged_sort="true",
          selective_host_offload="true",
          host_offload_param_patterns="['.*embed.*']",
          moe_dropless_fallback="step",
      )
    cfg_non_emb = _tiny_m3_config(
        out_dir,
        scan_layers="false",
        selective_host_offload="true",
        host_offload_param_patterns="['.*wi_0/kernel$']",
    )
    with jax.set_mesh(mesh_1d):
      with self.assertRaisesRegex(ValueError, "nnx.Embed"):
        model_creation_utils.create_model(cfg_non_emb, mesh_1d, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    cfg_sp = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        selective_host_offload="true",
        selective_host_offload_optimizer_mode="host_compute",
    )
    mesh_sp = mesh_from_config(cfg_sp)
    sh_sp = sharding_from_config(cfg_sp)
    with jax.set_mesh(mesh_sp):
      m_sp, _ = loop.setup_model(cfg_sp, mesh_sp, sh_sp)
      opt_sp = m3_offload.make_optimizer(
          m_sp,
          mt_optimizers.skip_step_on_spikes(optax.adamw(1e-2), interval=4, scaling_factor=1.0),
          cfg_sp,
      )
      b_sp = loop.make_synthetic_batch(cfg_sp, mesh_sp, sh_sp, jax.random.key(99))
      _, grads_sp = nnx.jit(lambda m, b: m3_offload.value_and_grad(loop.loss_fn, m, b))(m_sp, b_sp)
      ple_var = [
          v for p, v in nnx.to_flat_state(nnx.state(m_sp, nnx.Param)) if p[-2:] == ("ngram_embedding", "embedding")
      ][0]
      p_before = np.asarray(ple_var[...]).copy()
      opt_sp.update(m_sp, grads_sp, loss=jnp.float32(jnp.nan), grad_norm=jnp.float32(1.0))
      self.assertTrue(bool(opt_sp.opt_state["is_skipped"][...]))
      np.testing.assert_array_equal(np.asarray(ple_var[...]), p_before)

      # Verify optimizer_memory_host_offload=True + skip_step_on_spikes preserves 1D rolling buffers & finite-spike skip.
      cfg_sp_opt = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          selective_host_offload="true",
          optimizer_memory_host_offload="true",
          selective_host_offload_optimizer_mode="host_compute",
      )
      m_sp2, _ = loop.setup_model(cfg_sp_opt, mesh_sp, sh_sp)
      opt_sp2 = m3_offload.make_optimizer(
          m_sp2,
          mt_optimizers.skip_step_on_spikes(optax.adamw(1e-2), interval=4, scaling_factor=1.0),
          cfg_sp_opt,
      )
      self.assertEqual(opt_sp2.opt_state["losses"][...].shape, (4,))
      self.assertEqual(opt_sp2.opt_state["losses"].get_metadata("memory_kind"), "device")
      for _ in range(4):
        opt_sp2.update(m_sp2, grads_sp, loss=jnp.float32(1.0), grad_norm=jnp.float32(1.0))
      self.assertEqual(opt_sp2.opt_state["losses"][...].shape, (4,))
      self.assertFalse(bool(opt_sp2.opt_state["is_skipped"][...]))
      ple_var2 = [
          v for p, v in nnx.to_flat_state(nnx.state(m_sp2, nnx.Param)) if p[-2:] == ("ngram_embedding", "embedding")
      ][0]
      p2_before = np.asarray(ple_var2[...]).copy()
      # Finite spike (100.0 >> mean 1.0 + 1.0 * std 0.0) must skip on BOTH device and host-offloaded tables!
      opt_sp2.update(m_sp2, grads_sp, loss=jnp.float32(100.0), grad_norm=jnp.float32(1.0))
      self.assertTrue(bool(opt_sp2.opt_state["is_skipped"][...]))
      np.testing.assert_array_equal(np.asarray(ple_var2[...]), p2_before)

      # Verify scanned PLE (`has_custom_ids_fn`) + MTP multi-call `token_embedder` recapture never leaks scan tracers.
      cfg_both = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="1",
          scan_layers="true",
          selective_host_offload="true",
          host_offload_param_patterns="['.*/embedding$']",
      )
      m_both, _ = loop.setup_model(cfg_both, mesh_sp, sh_sp)
      b_both = loop.make_synthetic_batch(cfg_both, mesh_sp, sh_sp, jax.random.key(77))
      loss_both, grads_both = nnx.jit(lambda m, b: m3_offload.value_and_grad(loop.loss_fn, m, b))(m_both, b_both)
      self.assertTrue(bool(jnp.isfinite(loss_both)))
      flat_g_both = dict(nnx.to_flat_state(grads_both))
      self.assertIsInstance(
          flat_g_both[("decoder", "ple", "ple_embedding", "ngram_embedding", "embedding")], m3_offload.SparseHostGrad
      )

    ids = jnp.array([[2, 2, 2, 5]], dtype=jnp.int32)
    dg = jnp.ones((1, 4, 8), dtype=jnp.float32) * 1.5
    sparse = m3_offload.SparseHostGrad(ids, dg, table_shape=(16, 8), table_dtype=jnp.float32)
    diag = legacy_train.step_diagnostics({"emb": sparse}, None, [], [])
    np.testing.assert_allclose(float(diag["diag/max_abs_grad"]), 4.5, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ("host_compute", "host_compute"),
      ("streamed_chunks", "streamed_chunks"),
      ("lazy_sparse", "lazy_sparse"),
  )
  def test_skip_step_on_spikes_legacy_train_modes(self, mode: str):
    """Verifies `skip_step_on_spikes=True` in `train.py` executes both non-skipped and skipped steps."""
    out_dir = self.create_tempdir().full_path
    cfg_tr = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        dataset_type="synthetic",
        skip_step_on_spikes="true",
        skip_step_interval="4",
        skip_step_scaling_factor="-100.0",
        selective_host_offload="true",
        selective_host_offload_optimizer_mode=mode,
    )
    mesh_tr = mesh_from_config(cfg_tr)
    with jax.set_mesh(mesh_tr):
      _, _, state_mesh_shardings, _, mesh_tr, _, _, _, _, _, train_state = train_utils.setup_train_loop(
          cfg_tr, None, mesh=mesh_tr
      )
      jit_model, state = nnx.split(train_state)
      orig_host_sqnorm = m3_offload._host_array_sqnorm  # pylint: disable=protected-access
      fallback_sqnorm_calls = []

      def _spy_host_sqnorm(*args, **kwargs):
        fallback_sqnorm_calls.append(1)
        return orig_host_sqnorm(*args, **kwargs)

      with mock.patch.object(m3_offload, "_host_array_sqnorm", side_effect=_spy_host_sqnorm):
        p_step, _ = train_utils.jit_train_and_eval_step(
            cfg_tr, jit_model, mesh_tr, state, state_mesh_shardings, legacy_train.train_step
        )
      self.assertEmpty(fallback_sqnorm_calls)
      bs = cfg_tr.global_batch_size_to_train_on
      data_sh = sharding_utils.get_input_data_sharding(cfg_tr, mesh_tr, rules=cfg_tr.logical_axis_rules)

      def _make_batch(seed: int):
        toks = jax.random.randint(jax.random.key(seed), (bs, 8), 0, cfg_tr.vocab_size, dtype=jnp.int32)
        pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (bs, 8))
        segs = jnp.ones((bs, 8), dtype=jnp.int32)
        return jax.device_put(
            {
                "inputs": toks,
                "targets": toks,
                "inputs_position": pos,
                "targets_position": pos,
                "inputs_segmentation": segs,
                "targets_segmentation": segs,
            },
            data_sh,
        )

      # Step 0 is inside warmup (count+1=1 < min_history=2 -> not skipped); step 2 has std > 0 and
      # scaling_factor=-100.0 -> guaranteed skipped while keeping host parameters bit-exact.
      state, m0 = p_step(state, _make_batch(10))
      self.assertEqual(float(m0["scalar"]["optim/step_skipped"]), 0.0)
      if mode == "lazy_sparse":
        self.assertNotIn("learning/param_norm", m0["scalar"])
      else:
        self.assertIn("learning/param_norm", m0["scalar"])
      state, _ = p_step(state, _make_batch(11))

      def _host_param_np(st):
        merged = nnx.merge(jit_model, st)
        leaves = [
            v.get_raw_value()
            for _, v in nnx.to_flat_state(nnx.state(merged.model, nnx.Param))
            if isinstance(v, m3_offload.HostOffloadedParam)
        ]
        self.assertNotEmpty(leaves)
        return np.asarray(leaves[0]).copy()

      ple_before = _host_param_np(state)
      state, m2 = p_step(state, _make_batch(12))
      self.assertEqual(float(m2["scalar"]["optim/step_skipped"]), 1.0)
      ple_after = _host_param_np(state)
      np.testing.assert_array_equal(ple_after, ple_before)

  @pytest.mark.tpu_backend
  def test_tpu7x_aot_legacy_train_and_eval_auto_sharding_zero_megascale(self):
    """Verifies `train.py` `train_step` and `eval_step` under `shard_mode=auto` emit 0 MegaScale ops."""
    self.enterContext(mock.patch.dict(os.environ, {"ALLOW_MULTIPLE_LIBTPU_LOAD": "1"}))
    out_dir = self.create_tempdir().full_path
    cfg_tr = _qwen38_mini_config(
        out_dir,
        compile_topology="tpu7x-8",
        compile_topology_num_slices="1",
        shard_mode="auto",
        ici_fsdp_parallelism="4",
        ici_expert_parallelism="2",
        per_device_batch_size="1",
        max_target_length="16",
        mtp_num_layers="0",
        scan_layers="true",
        log_step_diagnostics="true",
        selective_host_offload="true",
    )
    topology_mesh = train_compile.get_topology_mesh(cfg_tr)
    with jax.set_mesh(topology_mesh):
      (
          shaped_train_args,
          shaped_train_kwargs,
          state_mesh_shardings,
          _,
          jit_model,
      ) = train_compile.get_shaped_inputs(topology_mesh, cfg_tr)
      state, shaped_batch = shaped_train_args
      data_sharding = sharding_utils.get_input_data_sharding(cfg_tr, topology_mesh, rules=cfg_tr.logical_axis_rules)
      p_train = train_utils.jit_train_step(
          cfg_tr,
          jit_model,
          state,
          state_mesh_shardings,
          data_sharding,
          legacy_train.train_step,
          None,
          mesh=topology_mesh,
      )
      p_eval = train_utils.jit_eval_step(cfg_tr, jit_model, state_mesh_shardings, data_sharding, legacy_train.eval_step)
      train_hlo = p_train.lower(state, shaped_batch, **shaped_train_kwargs).compile().as_text() or ""
      eval_hlo = p_eval.lower(state, shaped_batch).compile().as_text() or ""
    self.assertNotIn("MegaScaleCollective", train_hlo)
    self.assertNotIn("MegaScaleCollective", eval_hlo)

  def test_skip_step_on_spikes_maxtext_training_engine(self):
    """Verifies `skip_step_on_spikes=True` in `MaxTextTrainingEngine` under selective host offload."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is None:
      self.skipTest("tunix not installed in this environment")
    from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    cfg_eng = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        skip_step_on_spikes="true",
        selective_host_offload="true",
    )
    eng = maxtext_engine.MaxTextTrainingEngine(cfg_eng, mesh=mesh_from_config(cfg_eng))
    bs = cfg_eng.global_batch_size_to_train_on
    for step in range(2):
      toks = jax.random.randint(jax.random.key(30 + step), (bs, 8), 0, cfg_eng.vocab_size, dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (bs, 8))
      seg = jnp.ones((bs, 8), dtype=jnp.int32)
      batch = {
          "inputs": toks,
          "targets": jnp.roll(toks, -1, axis=-1),
          "inputs_position": pos,
          "targets_position": pos,
          "inputs_segmentation": seg,
          "targets_segmentation": seg,
      }
      if step == 0:
        eng.compile(batch)
      eng.fwd_bwd(batch)
      eng.update()
      m_out = eng.get_metrics()
      lk = next(iter(m_out.weighted_metrics))
      self.assertTrue(
          np.isfinite(
              float(jnp.sum(m_out.weighted_metrics[lk].unreduced_sum) / jnp.sum(m_out.weighted_metrics[lk].denominator))
          )
      )

  @parameterized.named_parameters(
      ("host_compute", "host_compute"),
      ("streamed_chunks", "streamed_chunks"),
      ("lazy_sparse", "lazy_sparse"),
  )
  def test_repeated_host_offloaded_embed_mtp_and_input_tokens_key(self, mode: str):
    """Verifies repeated `HostOffloadedEmbed` MTP calls and `'input_tokens'` extraction."""
    out_dir = self.create_tempdir().full_path

    def _mtp_loss(model: nnx.Module, batch: dict) -> jax.Array:
      toks = batch.get("inputs", batch.get("input_tokens"))
      seg = batch.get("inputs_segmentation", batch.get("segment_ids"))
      logits, mtp_logits = model(
          toks,
          decoder_positions=batch["inputs_position"],
          decoder_segment_ids=seg,
          return_mtp=True,
      )
      l_main = jnp.mean(jax.nn.log_softmax(logits, axis=-1)[..., 0])
      l_mtp = jnp.mean(jax.nn.log_softmax(mtp_logits, axis=-1)[..., 1])
      return -(l_main + 0.5 * l_mtp)

    cfg_on = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        scan_layers="true",
        mtp_num_layers="1",
        selective_host_offload="true",
        host_offload_param_patterns="['^decoder/token_embedder/embedding$']",
        selective_host_offload_optimizer_mode=mode,
    )
    mesh = mesh_from_config(cfg_on)
    sharding = sharding_from_config(cfg_on)
    num_devs = len(jax.devices())
    bs = int(cfg_on.per_device_batch_size) * num_devs
    ref_cache = type(self)._mtp_ref_cache  # pylint: disable=protected-access

    with jax.set_mesh(mesh):
      toks = jax.random.randint(jax.random.key(77), (bs, 8), 1, cfg_on.vocab_size, dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (bs, 8))
      seg = jnp.ones((bs, 8), dtype=jnp.int32)
      batch = {"inputs": toks, "inputs_position": pos, "inputs_segmentation": seg}

      if num_devs not in ref_cache:
        cfg_ref = _qwen38_mini_config(
            out_dir,
            max_target_length="8",
            scan_layers="true",
            mtp_num_layers="1",
            selective_host_offload="false",
        )
        m_ref, opt_ref = loop.setup_model(cfg_ref, mesh, sharding)
        toks_const = jnp.ones_like(toks) * 3
        jit_mtp_loss = nnx.jit(_mtp_loss)
        l_const = float(jit_mtp_loss(m_ref, {"inputs": toks_const, "inputs_position": pos, "inputs_segmentation": seg}))
        l_ref_arr, g_ref = nnx.jit(nnx.value_and_grad(_mtp_loss))(m_ref, batch)
        l_ref = float(l_ref_arr)
        g_ref_map = {tuple(p): np.asarray(v.get_raw_value()) for p, v in nnx.to_flat_state(g_ref)}
        nnx.jit(lambda m, o, g: o.update(m, g))(m_ref, opt_ref, g_ref)
        p_ref_after = {tuple(p): np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(m_ref, nnx.Param))}
        ref_cache[num_devs] = (l_const, l_ref, g_ref_map, p_ref_after)
      l_const, l_ref, g_ref_map, p_ref_after = ref_cache[num_devs]
      self.assertGreater(abs(l_ref - l_const), 1e-3)

      next_tok = np.pad(np.asarray(toks)[:, 1:], ((0, 0), (0, 1)))
      touched_rows = sorted(set(np.asarray(toks).reshape(-1).tolist()) | set(next_tok.reshape(-1).tolist()))
      emb_path = ("decoder", "token_embedder", "embedding")

      m_on, opt_on = loop.setup_model(cfg_on, mesh, sharding)
      call_batch = {"input_tokens": toks, "inputs_position": pos, "segment_ids": seg} if mode == "host_compute" else batch
      l_on, g_on = nnx.jit(lambda m, b: m3_offload.value_and_grad(_mtp_loss, m, b))(m_on, call_batch)
      np.testing.assert_allclose(float(l_on), l_ref, rtol=1e-6, atol=1e-6, err_msg=f"MTP loss {mode}")
      g_on_map = {tuple(p): (v.get_raw_value() if isinstance(v, nnx.Variable) else v) for p, v in nnx.to_flat_state(g_on)}
      for path, go in g_ref_map.items():
        gn = g_on_map[path]
        if isinstance(gn, m3_offload.SparseHostGrad):
          gn = m3_offload.materialize_sparse_grad(gn)
        np.testing.assert_allclose(np.asarray(gn), go, rtol=1e-5, atol=1e-5, err_msg=f"MTP grad {path} ({mode})")
      nnx.jit(lambda m, o, g: o.update(m, g))(m_on, opt_on, g_on)
      p_on_after = {tuple(p): np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(m_on, nnx.Param))}
      if mode == "lazy_sparse":
        np.testing.assert_allclose(
            p_on_after[emb_path][touched_rows],
            p_ref_after[emb_path][touched_rows],
            rtol=1e-5,
            atol=1e-5,
        )
      else:
        np.testing.assert_allclose(p_on_after[emb_path], p_ref_after[emb_path], rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(("qwen38_auto", True), ("tiny_explicit", False))
  def test_checkpoint_dict_and_optimizer_init_never_stage_host_tables_to_device(self, use_qwen38_auto: bool):
    """Verifies `loop.setup_model` and checkpoint save/restore keep host tables and moments off HBM."""
    out_dir = self.create_tempdir().full_path
    orig_h2d = m3_offload._host_to_device_weight  # pylint: disable=protected-access
    orig_zeros_like = jnp.zeros_like
    eager_host_zeros: list[tuple[int, ...]] = []

    def _checked_zeros_like(arr, *args, **kwargs):
      if not isinstance(arr, jax.core.Tracer) and isinstance(arr, jax.Array):
        if getattr(getattr(arr, "sharding", None), "memory_kind", None) == "pinned_host":
          eager_host_zeros.append(tuple(arr.shape))
      return orig_zeros_like(arr, *args, **kwargs)

    if use_qwen38_auto:
      cfg = _qwen38_mini_config(
          out_dir,
          shard_mode="auto",
          scan_layers="true",
          dtype="bfloat16",
          weight_dtype="float32",
          selective_host_offload="true",
      )
      leaf_sub = "ngram_embedding"
    else:
      cfg = _tiny_m3_config(
          out_dir,
          shard_mode="explicit",
          scan_layers="true",
          dtype="bfloat16",
          weight_dtype="float32",
          selective_host_offload="true",
          host_offload_param_patterns="['.*embed.*']",
      )
      leaf_sub = "token_embedder"

    mesh = mesh_from_config(cfg)
    sharding = sharding_from_config(cfg)
    with warnings.catch_warnings(record=True) as caught_warns:
      warnings.simplefilter("always")
      with (
          jax.set_mesh(mesh),
          nnx.use_eager_sharding(True),
          mock.patch.object(m3_offload, "_host_to_device_weight", wraps=orig_h2d) as spy_init,
          mock.patch.object(jnp, "zeros_like", side_effect=_checked_zeros_like),
      ):
        model, opt = loop.setup_model(cfg, mesh, sharding)
        self.assertEqual(spy_init.call_count, 0)
        self.assertEmpty(eager_host_zeros)

        hook_warns = [str(w.message) for w in caught_warns if "Variable hooks are deprecated" in str(w.message)]
        self.assertEmpty(hook_warns)
        for _, var in nnx.to_flat_state(nnx.state(model, nnx.Param)):
          self.assertIsNone(var.get_metadata("on_get_value", default=None))
        for leaf in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
          if isinstance(leaf, nnx.Variable):
            self.assertIsNone(leaf.get_metadata("on_get_value", default=None))

        ts = train_state_nnx.TrainStateNNX(model, opt)
        with mock.patch.object(m3_offload, "_host_to_device_weight", wraps=orig_h2d) as spy_ckpt:
          ckpt_dict = train_state_nnx.to_checkpoint_dict(nnx.state(ts))
          self.assertEqual(spy_ckpt.call_count, 0)

        ple_leaves = [
            val
            for key_path, val in jax.tree_util.tree_leaves_with_path(ckpt_dict)
            if leaf_sub in "/".join(str(m3_offload._key_value(k)) for k in key_path)  # pylint: disable=protected-access
        ]
        self.assertLen(ple_leaves, 3)
        for lf in ple_leaves:
          self.assertEqual(lf.sharding.memory_kind, "pinned_host")

        for as_numpy in (False, True):
          linen_state, _, _ = train_state_nnx.split_for_checkpoint(nnx.state(ts))
          src_dict = {k: v for k, v in ckpt_dict.items() if k != "nnx_aux"}
          if as_numpy:
            src_dict = jax.tree.map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, src_dict)
          restored_nested = train_state_nnx.from_linen_checkpoint_dict(src_dict)
          with mock.patch.object(m3_offload, "_host_to_device_weight", wraps=orig_h2d) as spy_restore:
            nnx.replace_by_pure_dict(linen_state, restored_nested)
            self.assertEqual(spy_restore.call_count, 0)
          nnx.update(ts, linen_state)
          for path, var in nnx.to_flat_state(nnx.state(ts.model, nnx.Param)):
            if leaf_sub in "/".join(str(x) for x in path):
              self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")
              self.assertEqual(var.get_value().sharding.memory_kind, "pinned_host")
          for path, leaf in jax.tree_util.tree_leaves_with_path(
              ts.optimizer.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
          ):
            key_s = "/".join(str(m3_offload._key_value(k)) for k in path)  # pylint: disable=protected-access
            if leaf_sub in key_s and getattr(leaf.get_raw_value(), "ndim", 0) == 2:
              self.assertEqual(leaf.get_raw_value().sharding.memory_kind, "pinned_host")

  def test_pretrain_and_engine_init_never_stage_host_tables_to_device(self):
    """Verifies `train.py` `setup_train_loop` and `MaxTextTrainingEngine.__init__` never stage host tables."""
    out_dir = self.create_tempdir().full_path
    orig_h2d = m3_offload._host_to_device_weight  # pylint: disable=protected-access
    orig_zeros_like = jnp.zeros_like
    eager_host_zeros: list[tuple[int, ...]] = []

    def _checked_zeros_like(arr, *args, **kwargs):
      if not isinstance(arr, jax.core.Tracer) and isinstance(arr, jax.Array):
        if getattr(getattr(arr, "sharding", None), "memory_kind", None) == "pinned_host":
          eager_host_zeros.append(tuple(arr.shape))
      return orig_zeros_like(arr, *args, **kwargs)

    cfg_tr = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        dtype="bfloat16",
        weight_dtype="float32",
        selective_host_offload="true",
    )
    with (
        nnx.use_eager_sharding(True),
        mock.patch.object(m3_offload, "_host_to_device_weight", wraps=orig_h2d) as spy_tr,
        mock.patch.object(jnp, "zeros_like", side_effect=_checked_zeros_like),
    ):
      *_, tr_state = train_utils.setup_train_loop(cfg_tr, None)
      self.assertEqual(spy_tr.call_count, 0)
      self.assertEmpty(eager_host_zeros)
      ckpt_tr = train_state_nnx.to_checkpoint_dict(tr_state)
      self.assertEqual(spy_tr.call_count, 0)
      ple_tr = [
          val
          for key_path, val in jax.tree_util.tree_leaves_with_path(ckpt_tr)
          if "ngram_embedding"
          in "/".join(str(m3_offload._key_value(k)) for k in key_path)  # pylint: disable=protected-access
      ]
      self.assertLen(ple_tr, 3)
      for lf in ple_tr:
        self.assertEqual(lf.sharding.memory_kind, "pinned_host")

      import importlib.util  # pylint: disable=import-outside-toplevel

      if importlib.util.find_spec("tunix") is not None:
        from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

        eng_mesh = mesh_from_config(cfg_tr)
        with jax.set_mesh(eng_mesh):
          eng = maxtext_engine.MaxTextTrainingEngine(cfg_tr, mesh=eng_mesh)
          self.assertEqual(spy_tr.call_count, 0)
          self.assertEmpty(eager_host_zeros)
          del eng

  def test_tied_explicit_sharding_forward_vocab_tiling_and_attend_parity(self):
    """Verifies tied Qwen3 explicit forward (`__call__`, `logits_from_hidden_states_for_vocab_tiling`, `attend`)."""
    out_dir = self.create_tempdir().full_path
    cfg_tied_off = _tiny_m3_config(out_dir, shard_mode="explicit", scan_layers="true", selective_host_offload="false")
    cfg_tied_on = _tiny_m3_config(
        out_dir,
        shard_mode="explicit",
        scan_layers="true",
        selective_host_offload="true",
        host_offload_param_patterns="['.*embed.*']",
    )
    t_mesh = mesh_from_config(cfg_tied_off)
    t_sharding = sharding_from_config(cfg_tied_off)
    with jax.set_mesh(t_mesh), nnx.use_eager_sharding(True), jax.default_matmul_precision("highest"):
      m_off, _ = loop.setup_model(cfg_tied_off, t_mesh, t_sharding)
      m_on, _ = loop.setup_model(cfg_tied_on, t_mesh, t_sharding)
      self.assertNotIsInstance(m_on, m3_offload.HostOffloadedParameters)
      self.assertIsInstance(m_on.token_embedder, m3_offload.HostOffloadedEmbed)
      self.assertIsInstance(m_on.token_embedder.embedding, m3_offload.HostOffloadedParam)
      self.assertEqual(m_on.token_embedder.embedding.get_value().sharding.memory_kind, "pinned_host")
      self.assertEqual(m_on.token_embedder.embedding[...].sharding.memory_kind, "device")
      batch = loop.make_synthetic_batch(cfg_tied_off, t_mesh, t_sharding, jax.random.key(9))
      np.testing.assert_allclose(
          np.asarray(m_on(batch["inputs"], batch["positions"])),
          np.asarray(m_off(batch["inputs"], batch["positions"])),
          rtol=1e-5,
          atol=1e-5,
      )
      hidden = jnp.ones((batch["inputs"].shape[0], 4, int(cfg_tied_off.base_emb_dim)), dtype=jnp.float32)
      hidden = jax.device_put(
          hidden,
          NamedSharding(t_mesh, t_sharding("embedded_inputs", ("batch", "norm_length", "embed"), TensorType.ACTIVATION)),
      )
      np.testing.assert_allclose(
          np.asarray(m_on.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic=True)),
          np.asarray(m_off.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic=True)),
          rtol=1e-5,
          atol=1e-5,
      )
      np.testing.assert_allclose(
          np.asarray(m_on.token_embedder.attend(hidden)),
          np.asarray(m_off.token_embedder.attend(hidden)),
          rtol=1e-5,
          atol=1e-5,
      )

  @pytest.mark.tpu_backend
  def test_tpu7x_aot_explicit_sharding_loop_eval_shape_preserves_pinned_host(self):
    """Verifies `nnx.eval_shape(loop.setup_model)` under `shard_mode=explicit` keeps `pinned_host` and compiles."""
    self.enterContext(mock.patch.dict(os.environ, {"ALLOW_MULTIPLE_LIBTPU_LOAD": "1"}))
    out_dir = self.create_tempdir().full_path
    topo = _get_tpu7x8_topology()

    def _compile_explicit(selective: bool, opt_mode: str = "host_compute"):
      cfg = _tiny_m3_config(
          out_dir,
          compile_topology="tpu7x-8",
          shard_mode="explicit",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          base_emb_dim="128",
          base_mlp_dim="256",
          head_dim="32",
          vocab_size="4096",
          scan_layers="true",
          dtype="bfloat16",
          weight_dtype="bfloat16",
          selective_host_offload=str(selective).lower(),
          host_offload_param_patterns="['.*embed.*']" if selective else "[]",
          selective_host_offload_optimizer_mode=opt_mode,
          selective_host_offload_num_chunks="2",
      )
      tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
      self.assertTrue(all(ax_t == AxisType.Explicit for ax_t in tpu_mesh.axis_types))
      sharding = sharding_from_config(cfg)
      with jax.set_mesh(tpu_mesh), nnx.use_eager_sharding(True):
        abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh, sharding))
        if selective:
          emb_sds = abs_model.token_embedder.embedding.get_raw_value()
          self.assertEqual(emb_sds.sharding.memory_kind, "pinned_host")
          sel_paths = m3_offload.selected_host_param_paths(abs_model)
          host_opt_count = 0
          for path, leaf in jax.tree_util.tree_leaves_with_path(
              abs_opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
          ):
            raw = leaf.get_raw_value()
            if getattr(raw, "ndim", 0) == 0:
              continue
            key_t = tuple(m3_offload._key_value(k) for k in path)  # pylint: disable=protected-access
            if m3_offload._is_matching_opt_leaf(key_t, sel_paths) is not None:  # pylint: disable=protected-access
              self.assertEqual(raw.sharding.memory_kind, "pinned_host")
              host_opt_count += 1
          self.assertEqual(host_opt_count, 2)
        data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
        data_sharding = NamedSharding(tpu_mesh, data_spec)
        batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
        batch = {
            "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
        }
        compiled = loop.train_step.lower(abs_model, abs_opt, batch).compile()
        self.assertNotIn("MegaScaleCollective", compiled.as_text())
        return compiled.memory_analysis()

    mem_off = _compile_explicit(False)
    for mode in ("host_compute", "streamed_chunks"):
      mem_on = _compile_explicit(True, mode)
      self.assertLess(mem_on.argument_size_in_bytes, mem_off.argument_size_in_bytes)

  def test_pretrain_zero1_excludes_selectively_offloaded_leaves_and_anchors_suffix_matching(self):
    """Verifies `get_abstract_state_nnx` ZeRO-1 skips `pinned_host` leaves and `_is_matching_opt_leaf` anchors paths."""
    # 1. Path-boundary check: nested `("token_embedder", "embedding")` must NOT match top-level `("embedding",)`.
    self.assertIsNone(
        m3_offload._is_matching_opt_leaf(  # pylint: disable=protected-access
            (0, "mu", "token_embedder", "embedding"),
            frozenset({("embedding",)}),
        )
    )
    self.assertEqual(
        m3_offload._is_matching_opt_leaf(  # pylint: disable=protected-access
            (0, "mu", "embedding"),
            frozenset({("embedding",)}),
        ),
        ("embedding",),
    )

    class _NestedSuffixModel(nnx.Module):
      """Model with both top-level `embedding` (selected) and nested `token_embedder.embedding` (unselected)."""

      def __init__(self):
        base_emb = m3_offload.to_memory(nnx.Param(jnp.ones((4, 4), dtype=jnp.float32)), "pinned_host")
        self.embedding = m3_offload._to_host_offloaded_param(base_emb)  # pylint: disable=protected-access
        self.token_embedder = nnx.Embed(num_embeddings=4, features=4, rngs=nnx.Rngs(0))

      def __call__(self, ids: jax.Array) -> jax.Array:
        w = m3_offload._host_to_device_weight(self.embedding.get_raw_value())  # pylint: disable=protected-access
        return w[ids] + self.token_embedder(ids)

    suffix_model = _NestedSuffixModel()
    suffix_opt = m3_offload.HostOffloadedOptimizer(
        suffix_model,
        optax.adamw(1e-2),
        wrt=nnx.Param,
        offload_opt_state=False,
        selected_param_paths=frozenset({("embedding",)}),
    )
    ids = jnp.array([[0, 1]], dtype=jnp.int32)
    grads = nnx.grad(lambda m: jnp.sum(m(ids)))(suffix_model)
    suffix_opt.update(suffix_model, grads)

  def test_pretrain_zero1_excludes_selectively_offloaded_leaves(self):
    """Verifies `get_abstract_state_nnx` ZeRO-1 skips `pinned_host` leaves under `selective_host_offload=True`."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 host devices")
    out_dir = self.create_tempdir().full_path
    cfg = _qwen38_mini_config(
        out_dir,
        ici_data_parallelism="4",
        ici_fsdp_parallelism="1",
        ici_expert_parallelism="1",
        shard_optimizer_over_data="true",
        selective_host_offload="true",
        dataset_type="synthetic",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      (
          _,
          _,
          state_mesh_shardings,
          _,
          mesh,
          _,
          data_iter,
          _,
          _,
          _,
          train_state,
      ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
      params_shardings, state_mesh_shardings = sharding_utils.maybe_update_params_sharding_with_opt(
          cfg, state_mesh_shardings
      )
      raw_param_shardings = [
          s.get_raw_value() if isinstance(s, nnx.Variable) else s
          for s in jax.tree_util.tree_leaves(
              params_shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, NamedSharding))
          )
      ]
      host_param_specs = [
          s.spec for s in raw_param_shardings if isinstance(s, NamedSharding) and s.memory_kind == "pinned_host"
      ]
      self.assertNotEmpty(host_param_specs)
      for spec in host_param_specs:
        self.assertNotIn("data", spec)

      raw_opt_shardings = [
          s.get_raw_value() if isinstance(s, nnx.Variable) else s
          for s in jax.tree_util.tree_leaves(
              state_mesh_shardings.optimizer, is_leaf=lambda x: isinstance(x, (nnx.Variable, NamedSharding))
          )
      ]
      host_opt_specs = [
          s.spec for s in raw_opt_shardings if isinstance(s, NamedSharding) and s.memory_kind == "pinned_host"
      ]
      self.assertNotEmpty(host_opt_specs)
      for spec in host_opt_specs:
        self.assertNotIn("data", spec)

      jit_model, state = nnx.split(train_state)
      p_step, _ = train_utils.jit_train_and_eval_step(
          cfg, jit_model, mesh, state, state_mesh_shardings, legacy_train.train_step, params_shardings=params_shardings
      )
      state = sharding_utils.reshard_train_state(state, state_mesh_shardings, cfg.enable_single_controller)
      _, metrics = p_step(state, next(data_iter))
      self.assertTrue(np.isfinite(float(metrics["scalar"]["learning/loss"])))

  def test_engine_auto_sharding_eval_and_fwd_only_avoid_full_table_all_gather(self):
    """Verifies MaxTextTrainingEngine eval_step and fwd_only preserve shard-local PLE gathers under ShardMode.AUTO."""
    out_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, r"lora\.enable_lora"):
      _qwen38_mini_config(
          out_dir,
          selective_host_offload="true",
          lora={"enable_lora": True, "lora_rank": 8, "lora_alpha": 16.0, "lora_module_path": ".*"},
      )

    @contextlib.contextmanager
    def _tunix_stub_or_real():
      import flax  # pylint: disable=import-outside-toplevel
      import importlib.util  # pylint: disable=import-outside-toplevel
      import sys  # pylint: disable=import-outside-toplevel
      import types as py_types  # pylint: disable=import-outside-toplevel

      if importlib.util.find_spec("tunix") is not None:
        from maxtext.training_engine import maxtext_engine as real_engine  # pylint: disable=import-outside-toplevel

        yield real_engine
        return

      te_pkg_name = f"{pyconfig.__name__.rsplit(".configs.", 1)[0]}.training_engine"
      tunix_mod = py_types.ModuleType("tunix")
      dist_rl = py_types.ModuleType("tunix.dist_rl")
      common = py_types.ModuleType("tunix.dist_rl.common")
      datatypes = py_types.ModuleType("tunix.dist_rl.common.datatypes")
      sft = py_types.ModuleType("tunix.sft")
      sft_utils = py_types.ModuleType("tunix.sft.utils")

      class _TrainerPayload:
        """TrainerPayload stub."""

      @flax.struct.dataclass
      class _RLTrainerPayload(_TrainerPayload):
        """RLTrainerPayload stub."""

      @flax.struct.dataclass
      class _LossOutput:
        """LossOutput stub."""

        primary_loss: object = None
        aux_metrics: object = None

      @flax.struct.dataclass
      class _WeightedMetric:
        """WeightedMetric stub."""

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
      tracked_keys = (*stub_modules, *(f"{te_pkg_name}.{attr}" for attr in engine_submodules))
      saved_sys = {key: sys.modules[key] for key in tracked_keys if key in sys.modules}
      pkg = sys.modules.get(te_pkg_name)
      saved_pkg_attrs = {attr: getattr(pkg, attr) for attr in engine_submodules if pkg is not None and hasattr(pkg, attr)}
      try:
        sys.modules.update(stub_modules)
        for attr in engine_submodules:
          sys.modules.pop(f"{te_pkg_name}.{attr}", None)
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
        pkg_after = sys.modules.get(te_pkg_name)
        if pkg_after is not None:
          for attr in engine_submodules:
            if attr in saved_pkg_attrs:
              setattr(pkg_after, attr, saved_pkg_attrs[attr])
            elif hasattr(pkg_after, attr):
              delattr(pkg_after, attr)

    cfg_on = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        ici_fsdp_parallelism="2",
        ici_expert_parallelism="2",
        ici_data_parallelism="1",
        shard_mode="auto",
        selective_host_offload="true",
    )
    cfg_off = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        ici_fsdp_parallelism="2",
        ici_expert_parallelism="2",
        ici_data_parallelism="1",
        shard_mode="auto",
        selective_host_offload="false",
    )
    mesh = mesh_from_config(cfg_on)
    bs = cfg_on.global_batch_size_to_train_on
    seq = cfg_on.max_target_length
    toks = jax.random.randint(jax.random.key(311), (bs, seq), 0, cfg_on.vocab_size, dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (bs, seq))
    seg = jnp.ones((bs, seq), dtype=jnp.int32)
    batch = {
        "inputs": toks,
        "targets": jnp.roll(toks, -1, axis=-1),
        "inputs_position": pos,
        "targets_position": pos,
        "inputs_segmentation": seg,
        "targets_segmentation": seg,
    }

    with _tunix_stub_or_real() as maxtext_engine:
      eng_on = maxtext_engine.MaxTextTrainingEngine(cfg_on, mesh=mesh)
      eng_off = maxtext_engine.MaxTextTrainingEngine(cfg_off, mesh=mesh)
      eng_on.compile(batch)
      eng_off.compile(batch)

      def _count_ple_table_ag(hlo_text: str) -> int:
        return sum(
            1
            for ln in hlo_text.splitlines()
            if "all-gather(" in ln and ("ple_embedding" in ln or "ngram_embedding" in ln or "1024,32]" in ln)
        )

      model_on = eng_on._live_model()  # pylint: disable=protected-access
      params_pure, rest_pure = eng_on._read_model_pure(model_on)  # pylint: disable=protected-access
      p_sh = jax.tree.map(eng_on._mesh_sharding, params_pure)  # pylint: disable=protected-access
      r_sh = jax.tree.map(eng_on._mesh_sharding, rest_pure)  # pylint: disable=protected-access
      dyn_b, stat_b = maxtext_engine._split_static_and_dynamic(  # pylint: disable=protected-access
          eng_on._prepare_batch(batch)  # pylint: disable=protected-access
      )

      # Negative control: without active_param_shardings, _eval_kernel emits an extra full-table [1024, 32] all-gather.
      with eng_on._sharding_ctx():  # pylint: disable=protected-access
        unguarded_eval_hlo = (
            jax.jit(
                lambda p, r, d: eng_on._eval_kernel(p, r, {**d, **stat_b}),  # pylint: disable=protected-access
                in_shardings=(p_sh, r_sh, eng_on._batch_data_shardings(dyn_b)),  # pylint: disable=protected-access
            )
            .lower(
                eng_on._mesh_avals(params_pure),  # pylint: disable=protected-access
                eng_on._mesh_avals(rest_pure),  # pylint: disable=protected-access
                eng_on._batch_avals(dyn_b),  # pylint: disable=protected-access
            )
            .compile()
            .as_text()
        )

      eval_hlo = eng_on._compiled_eval.as_text()  # pylint: disable=protected-access
      self.assertNotIn("MegaScaleCollective", eval_hlo)
      self.assertLess(
          _count_ple_table_ag(eval_hlo),
          _count_ple_table_ag(unguarded_eval_hlo),
          "Expected active_param_shardings in _compile_eval_for_batch to eliminate full-table PLE all-gather.",
      )
      ple_named_eval_ag = [
          ln
          for ln in eval_hlo.splitlines()
          if "all-gather(" in ln and "1024,32]" in ln and ("ple_embedding" in ln or "ngram_embedding" in ln)
      ]
      self.assertEmpty(ple_named_eval_ag)

      with eng_on.eval_context():
        eng_on.eval_step(batch)
      with eng_off.eval_context():
        eng_off.eval_step(batch)

      def _fwd_cb(mdl, pld):
        return mdl(pld["inputs"], pld["inputs_position"], pld["inputs_segmentation"])

      eng_on.with_fwd_only_fn(_fwd_cb)
      eng_off.with_fwd_only_fn(_fwd_cb)
      out_on = eng_on.fwd_only(batch)
      out_off = eng_off.fwd_only(batch)
      self.assertEqual(getattr(eng_on._compiled_fwd_only.__wrapped__, "__name__", ""), "fwd_only")  # pylint: disable=protected-access
      self.assertEqual(getattr(eng_off._compiled_fwd_only.__wrapped__, "__name__", ""), "fwd_only")  # pylint: disable=protected-access
      np.testing.assert_allclose(np.asarray(out_on), np.asarray(out_off), rtol=1e-5, atol=1e-5)

      prepped = jax.device_put(batch, eng_on._fwd_only_data_shardings(batch))  # pylint: disable=protected-access
      with eng_on._sharding_ctx():  # pylint: disable=protected-access
        unguarded_fwd_hlo = (
            jax.jit(
                eng_on._fwd_only_fn,  # pylint: disable=protected-access
                in_shardings=(
                    None,
                    p_sh,
                    r_sh,
                    eng_on._fwd_only_data_shardings(prepped),  # pylint: disable=protected-access
                ),
            )
            .lower(
                eng_on._model_graphdef,  # pylint: disable=protected-access
                eng_on._mesh_avals(params_pure),  # pylint: disable=protected-access
                eng_on._mesh_avals(rest_pure),  # pylint: disable=protected-access
                prepped,
            )
            .compile()
            .as_text()
        )
        fwd_hlo = (
            eng_on._compiled_fwd_only.lower(  # pylint: disable=protected-access
                eng_on._model_graphdef,  # pylint: disable=protected-access
                eng_on._mesh_avals(params_pure),  # pylint: disable=protected-access
                eng_on._mesh_avals(rest_pure),  # pylint: disable=protected-access
                prepped,
            )
            .compile()
            .as_text()
        )
      self.assertNotIn("MegaScaleCollective", fwd_hlo)
      self.assertLess(
          _count_ple_table_ag(fwd_hlo),
          _count_ple_table_ag(unguarded_fwd_hlo),
          "Expected active_param_shardings in _compile_fwd_only_for_batch to eliminate full-table PLE all-gather.",
      )
      ple_named_fwd_ag = [
          ln
          for ln in fwd_hlo.splitlines()
          if "all-gather(" in ln and "1024,32]" in ln and ("ple_embedding" in ln or "ngram_embedding" in ln)
      ]
      self.assertEmpty(ple_named_fwd_ag)

      # Frozen host embeddings via `trainable_parameters_mask` must emit exact zero host gradients
      # and exclude frozen tables from `gradient_norm` and weight updates.
      cfg_frz_on = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          ici_fsdp_parallelism="2",
          ici_expert_parallelism="2",
          ici_data_parallelism="1",
          shard_mode="auto",
          selective_host_offload="true",
          trainable_parameters_mask=["^(?!.*(ple_embedding|ngram_embedding)).*"],
      )
      cfg_frz_off = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          ici_fsdp_parallelism="2",
          ici_expert_parallelism="2",
          ici_data_parallelism="1",
          shard_mode="auto",
          selective_host_offload="false",
          trainable_parameters_mask=["^(?!.*(ple_embedding|ngram_embedding)).*"],
      )
      eng_frz_on = maxtext_engine.MaxTextTrainingEngine(cfg_frz_on, mesh=mesh)
      eng_frz_off = maxtext_engine.MaxTextTrainingEngine(cfg_frz_off, mesh=mesh)
      _, p_frz_off, _ = nnx.split(eng_frz_off.model, nnx.Param, ...)
      _, p_frz_on, _ = nnx.split(eng_frz_on.model, nnx.Param, ...)
      for path_t, var_off in nnx.to_flat_state(p_frz_off):
        var_on = dict(nnx.to_flat_state(p_frz_on))[path_t]
        sh_on = getattr(var_on.get_raw_value(), "sharding", None)
        if sh_on is not None:
          var_on.set_raw_value(jax.device_put(np.asarray(var_off.get_raw_value()), sh_on))
      eng_frz_on.compile(batch)
      eng_frz_off.compile(batch)
      flat_on_before = dict(nnx.to_flat_state(nnx.state(eng_frz_on.model, nnx.Param)))
      host_frz_paths = [p for p, v in flat_on_before.items() if isinstance(v, m3_offload.HostOffloadedParam)]
      self.assertNotEmpty(host_frz_paths)
      ple_before = {p: np.asarray(flat_on_before[p].get_raw_value()).copy() for p in host_frz_paths}
      eng_frz_on.fwd_bwd(batch)
      eng_frz_off.fwd_bwd(batch)
      flat_acc_grads = dict(nnx.to_flat_state(eng_frz_on._accumulated_grads))  # pylint: disable=protected-access
      for p in host_frz_paths:
        self.assertEqual(float(np.max(np.abs(np.asarray(flat_acc_grads[p].get_raw_value())))), 0.0)
      eng_frz_on.update()
      eng_frz_off.update()
      gn_on = float(np.asarray(eng_frz_on.get_metrics().scalar_metrics["gradient_norm"]).reshape(-1)[0])
      gn_off = float(np.asarray(eng_frz_off.get_metrics().scalar_metrics["gradient_norm"]).reshape(-1)[0])
      np.testing.assert_allclose(gn_on, gn_off, rtol=1e-5, atol=1e-5)
      flat_on_after = dict(nnx.to_flat_state(nnx.state(eng_frz_on.model, nnx.Param)))
      for p in host_frz_paths:
        np.testing.assert_array_equal(np.asarray(flat_on_after[p].get_raw_value()), ple_before[p])

  def test_traced_selective_offload_places_device_tracers_in_host_space_and_none_leaves_align(self):
    """Verifies traced `maybe_offload_parameters` moves `Space.Device` tracers to `Space.Host` and `None` leaves align."""
    out_dir = self.create_tempdir().full_path
    cfg = _qwen38_mini_config(
        out_dir,
        selective_host_offload="true",
        host_offload_param_patterns=[r"^emb/embedding$"],
    )

    class _MiniEmbedMod(nnx.Module):
      """Minimal module wrapping `nnx.Embed` for traced offload verification."""

      def __init__(self, table: jax.Array):
        self.emb = nnx.Embed(num_embeddings=8, features=4, rngs=nnx.Rngs(0))
        self.emb.embedding = nnx.Param(table)

    seen_spaces: list[object] = []

    @jax.jit
    def _trace_offload(x_dev: jax.Array) -> jax.Array:
      mod = _MiniEmbedMod(x_dev)
      m3_offload.maybe_offload_parameters(mod, cfg)
      raw = mod.emb.embedding.get_raw_value()
      seen_spaces.append(getattr(getattr(raw, "aval", None), "memory_space", None))
      return mod.emb(jnp.array([[1, 2]], dtype=jnp.int32))

    out = _trace_offload(jnp.ones((8, 4), dtype=jnp.float32))
    self.assertEqual(out.shape, (1, 2, 4))
    self.assertEqual(seen_spaces, [jax.memory.Space.Host])

    mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))
    host_sh = NamedSharding(mesh, P("fsdp", None), memory_kind="pinned_host")
    dev_sh = NamedSharding(mesh, P("fsdp", None))
    host_arr = jax.device_put(np.arange(8, dtype=np.float32).reshape(4, 2), host_sh)
    dev_arr = jax.device_put(np.ones((4, 2), dtype=np.float32), dev_sh)
    host_var = m3_offload.HostOffloadedParam(host_arr, memory_kind="pinned_host")
    mixed_tree = {"none_first": None, "host": host_var, "dev": dev_arr}
    sh_tree_none_hint = {"none_first": dev_sh, "host": None, "dev": dev_sh}
    expected_norm = float(np.sqrt(np.sum(np.arange(8, dtype=np.float32) ** 2) + 8.0))
    np.testing.assert_allclose(
        float(m3_offload.l2norm_pytree(mixed_tree, shardings=sh_tree_none_hint)),
        expected_norm,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(m3_offload.max_abs_grad_pytree(mixed_tree, shardings=sh_tree_none_hint)),
        7.0,
        rtol=1e-6,
        atol=1e-6,
    )
    with self.assertRaises(ValueError):
      m3_offload.l2norm_pytree(mixed_tree, shardings={"host": None})
    with self.assertRaises(ValueError):
      m3_offload.max_abs_grad_pytree(mixed_tree, shardings={"host": None})
    clipped, raw_n, clip_n = m3_offload.clip_grads_and_norms(mixed_tree, 2.0, shardings=sh_tree_none_hint)
    self.assertIsNone(clipped["none_first"])
    np.testing.assert_allclose(float(raw_n), expected_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(clip_n), 2.0, rtol=1e-6, atol=1e-6)
    zeroed = m3_offload.zero_tree_if(jnp.bool_(True), mixed_tree, shardings=sh_tree_none_hint)
    self.assertIsNone(zeroed["none_first"])
    self.assertEqual(float(np.max(np.abs(np.asarray(zeroed["host"].get_raw_value())))), 0.0)
    selected = m3_offload.select_tree_if(jnp.bool_(False), zeroed, mixed_tree, shardings=sh_tree_none_hint)
    self.assertIsNone(selected["none_first"])
    np.testing.assert_allclose(np.asarray(selected["host"].get_raw_value()), np.asarray(host_arr))

    # All-host-offloaded model (`dev_paths` empty) with `skip_step_on_spikes` + `pinned_host` spike buffers
    # must preserve `pinned_host` memory_kind on outer `st_probe` rolling buffers across jitted `update`.
    from maxtext.optimizers import optimizers as mt_optimizers  # pylint: disable=import-outside-toplevel

    cfg_all_host = _qwen38_mini_config(
        out_dir,
        selective_host_offload="true",
        host_offload_param_patterns=[r"^emb/embedding$"],
        optimizer_memory_host_offload="true",
        skip_step_on_spikes="true",
    )
    with jax.set_mesh(mesh):
      mod_all_host = _MiniEmbedMod(jax.device_put(np.ones((8, 4), dtype=np.float32), dev_sh))
      m3_offload.maybe_offload_parameters(mod_all_host, cfg_all_host)
      base_tx = mt_optimizers.skip_step_on_spikes(optax.adamw(1e-3), interval=4, scaling_factor=2.0)
      opt_all_host = m3_offload.make_optimizer(mod_all_host, base_tx, cfg_all_host, wrt=nnx.Param)
      opt_all_host.opt_state = nnx.State(
          jax.tree.map(
              lambda v: (
                  v.replace(
                      jax.device_put(
                          v.get_raw_value(),
                          NamedSharding(mesh, P(), memory_kind="pinned_host"),
                      )
                  )
                  if isinstance(v, nnx.Variable)
                  and hasattr(v.get_raw_value(), "ndim")
                  and not m3_offload._is_host_leaf(v)  # pylint: disable=protected-access
                  else v
              ),
              opt_all_host.opt_state,
              is_leaf=lambda x: isinstance(x, nnx.Variable),
          )
      )
      gdef_ah, st_ah = nnx.split((mod_all_host, opt_all_host))
      st_ah_sh = jax.tree.map(lambda a: a.sharding, st_ah)

      def _step_all_host(st_in):
        m_in, o_in = nnx.merge(gdef_ah, st_in, copy=True)
        ids = jnp.array([[1, 2]], dtype=jnp.int32)
        (_, _), grads_in = m3_offload.value_and_grad(
            lambda m, t: (jnp.sum(m.emb(t) ** 2), None),
            m_in,
            ids,
            has_aux=True,
            input_tokens=ids,
        )
        o_in.update(m_in, grads_in, loss=jnp.float32(1.0), grad_norm=jnp.float32(1.0))
        _, st_out = nnx.split((m_in, o_in))
        return st_out

      st_next = jax.jit(_step_all_host, in_shardings=(st_ah_sh,))(st_ah)
      for leaf in jax.tree.leaves(st_next[1]["opt_state"]):
        if hasattr(leaf, "ndim"):
          self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "pinned_host")

  def test_selective_plus_optimizer_host_offload_and_spikes_never_stages_host_tables(self):
    """Verifies `selective_host_offload` + `optimizer_memory_host_offload` + `skip_step_on_spikes` across drivers."""
    out_dir = self.create_tempdir().full_path
    cfg = _qwen38_mini_config(
        out_dir,
        max_target_length="8",
        mtp_num_layers="0",
        scan_layers="true",
        dataset_type="synthetic",
        selective_host_offload="true",
        optimizer_memory_host_offload="true",
        skip_step_on_spikes="true",
    )
    mesh = mesh_from_config(cfg)
    orig_jit = jax.jit
    staged_ple_to_dev = []

    def _spy_jit(fun, *args, **kwargs):
      out_sh = kwargs.get("out_shardings")
      if out_sh is not None:
        for kp, sh_leaf in jax.tree_util.tree_leaves_with_path(out_sh, is_leaf=lambda x: isinstance(x, NamedSharding)):
          if isinstance(sh_leaf, NamedSharding) and sh_leaf.memory_kind == "device":
            kp_str = "/".join(str(m3_offload._key_value(k)) for k in kp)  # pylint: disable=protected-access
            if "ple_embedding" in kp_str or "ngram_embedding" in kp_str:
              staged_ple_to_dev.append(kp_str)
      return orig_jit(fun, *args, **kwargs)

    with jax.set_mesh(mesh):
      with mock.patch.object(jax, "jit", side_effect=_spy_jit):
        _, _, state_sh, _, _, _, _, _, _, _, train_state = train_utils.setup_train_loop(cfg, None, mesh=mesh)
      self.assertEmpty(staged_ple_to_dev)
      # Both PLE moments and non-selected device-param moments reside in `pinned_host`, while scalar/rolling
      # spike bookkeeping (`is_skipped`, `loss_history`, `grad_norm_history`, `count`, `step`) stays on `device`.
      opt_kinds = {
          "/".join(str(m3_offload._key_value(k)) for k in kp): sh_leaf.memory_kind  # pylint: disable=protected-access
          for kp, sh_leaf in jax.tree_util.tree_leaves_with_path(
              state_sh.optimizer, is_leaf=lambda x: isinstance(x, NamedSharding)
          )
          if isinstance(sh_leaf, NamedSharding)
      }
      self.assertTrue(any("mu/" in k and v == "pinned_host" for k, v in opt_kinds.items()))
      self.assertEqual([v for k, v in opt_kinds.items() if "is_skipped" in k], ["device"])
      self.assertEqual([v for k, v in opt_kinds.items() if "opt_state/losses" in k], ["device"])
      jit_model, state = nnx.split(train_state)
      p_step, _ = train_utils.jit_train_and_eval_step(cfg, jit_model, mesh, state, state_sh, legacy_train.train_step)
      bs = cfg.global_batch_size_to_train_on
      data_sh = sharding_utils.get_input_data_sharding(cfg, mesh, rules=cfg.logical_axis_rules)
      toks = jax.random.randint(jax.random.key(77), (bs, 8), 0, cfg.vocab_size, dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (bs, 8))
      segs = jnp.ones((bs, 8), dtype=jnp.int32)
      batch = jax.device_put(
          {
              "inputs": toks,
              "targets": toks,
              "inputs_position": pos,
              "targets_position": pos,
              "inputs_segmentation": segs,
              "targets_segmentation": segs,
          },
          data_sh,
      )
      state, metrics = p_step(state, batch)
      self.assertTrue(np.isfinite(float(metrics["scalar"]["learning/loss"])))

  @pytest.mark.tpu_backend
  def test_tpu7x_aot_selective_plus_optimizer_host_offload_and_spikes_all_modes(self):
    """Verifies `tpu7x-8` AOT compiles all 3 modes with `optimizer_memory_host_offload` and `skip_step_on_spikes`."""
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    for mode in ("host_compute", "streamed_chunks", "lazy_sparse"):
      cfg = _qwen38_mini_config(
          out_dir,
          max_target_length="8",
          mtp_num_layers="0",
          scan_layers="true",
          compile_topology="tpu7x-8",
          compile_topology_num_slices="1",
          ici_fsdp_parallelism="4",
          ici_expert_parallelism="2",
          selective_host_offload="true",
          selective_host_offload_optimizer_mode=mode,
          optimizer_memory_host_offload="true",
          skip_step_on_spikes="true",
      )
      t_mesh = train_compile.get_topology_mesh(cfg)
      shaped_args, shaped_kw, state_sh, _, model = train_compile.get_shaped_inputs(t_mesh, cfg)
      p_sh, state_sh = sharding_utils.maybe_update_params_sharding_with_opt(cfg, state_sh)
      in_st_sh = sharding_utils.build_zero1_input_state_mesh_shardings(cfg, state_sh, p_sh)
      d_sh = sharding_utils.get_input_data_sharding(cfg, t_mesh)
      fn, in_s, out_s, stat_a, don_a = maxtext_utils.get_functional_train_with_signature(
          legacy_train.train_step, d_sh, in_st_sh, model, cfg, p_sh
      )
      compiled = train_compile.jit_and_compile(
          fn,
          shaped_args,
          shaped_kw,
          t_mesh,
          in_s,
          out_s,
          stat_a,
          don_a,
          cfg,
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
      )
      self.assertNotIn("MegaScaleCollective", compiled.as_text())


if __name__ == "__main__":
  absltest.main()

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

from unittest import mock

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
from maxtext.trainers.pre_train import train as legacy_train
from maxtext.training_engine import memory_report
from maxtext.utils import train_utils
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


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
        self.assertTrue(all(a.sharding.memory_kind == expected_opt_kind for a in opt_arrays))

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
        self.assertTrue(all(a.sharding.memory_kind == expected_opt_kind for a in post_opt_arrays))

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

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))

      def __call__(self, x):
        return x @ self.w[...]

    class _Cfg:
      parameter_memory_host_offload = True

    mod = m3_offload.maybe_offload_parameters(_MinimalModule(), _Cfg())
    with self.assertRaisesRegex(AttributeError, "logits_from_hidden_states_for_vocab_tiling"):
      mod.logits_from_hidden_states_for_vocab_tiling(jnp.ones((2, 2)))


class RuntimeHbmTelemetryTest(absltest.TestCase):
  """Tests `get_device_hbm_stats` and `loop._log_step` HBM reporting."""

  def test_cpu_and_empty_memory_stats_return_none(self):
    self.assertIsNone(m3_offload.get_device_hbm_stats())

    class _NoneStatsDevice:

      def memory_stats(self):
        return None

    class _IncompleteStatsDevice:

      def memory_stats(self):
        return {"bytes_reserved": 1024}

    class _NonAddressablePathwaysDevice:

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

  def test_eval_shape_preserves_pinned_host_sharding_for_aot_compile(self):
    out_dir = self.create_tempdir().full_path
    cfg = _tiny_m3_config(
        out_dir,
        optimizer_memory_host_offload="true",
        parameter_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, mesh))

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
      self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")
    for var in opt_vars:
      self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")

  def test_tpu7x_aot_compile_reduces_device_argument_and_kernel_device_peak(self):
    out_dir = self.create_tempdir().full_path
    import os  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    th = accelerator_to_spec_map.get_system_characteristics("tpu7x-8")
    topo = get_topology_desc(
        platform=th.platform,
        topology_name=th.topology_name,
        chip_config_name=th.chip_config_name,
        chips_per_host_bounds=th.chips_per_host_bounds,
        num_slices=1,
        wrap=th.wrap,
    )

    def _compile_step(offload_opt: bool, offload_param: bool = False):
      cfg = _tiny_m3_config(
          out_dir,
          compile_topology="tpu7x-8",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          optimizer_memory_host_offload=str(offload_opt).lower(),
          parameter_memory_host_offload=str(offload_param).lower(),
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
        lowered = loop.train_step.lower(abs_model, abs_opt, batch)
      compiled = lowered.compile()
      rows = memory_report.memory_report({"update": compiled}, {})
      return compiled.memory_analysis(), rows[0], memory_report.format_memory_report(rows)

    # Negative control: optimizer_memory_host_offload=False keeps all optimizer moments in HBM.
    mem_a, row_a, report_a = _compile_step(offload_opt=False, offload_param=False)
    mem_b, row_b, report_b = _compile_step(offload_opt=True, offload_param=False)
    _, row_c, _ = _compile_step(offload_opt=True, offload_param=True)

    self.assertEqual(mem_a.host_argument_size_in_bytes, 0)
    self.assertEqual(row_a.host_argument, 0)
    self.assertIn(memory_report.PEAK_LABEL, report_a)
    self.assertIn(memory_report.PEAK_LABEL, report_b)

    # Offloading FP32 AdamW moments moves ~2/3 of static state bytes from HBM to S(5) pinned_host
    # and reduces device-resident kernel peak (`KernelMemory.device_total`).
    offloaded_opt_bytes = row_a.argument - row_b.argument
    self.assertGreater(offloaded_opt_bytes, int(0.6 * row_a.argument))
    self.assertEqual(row_b.host_argument, offloaded_opt_bytes)
    self.assertEqual(mem_b.host_argument_size_in_bytes, offloaded_opt_bytes)
    self.assertLess(row_b.argument, row_a.argument)

    # Offloading parameters in addition to optimizer state further reduces device argument and resident bytes.
    self.assertLess(row_c.argument, row_b.argument)
    self.assertGreater(row_c.host_argument, row_b.host_argument)

  def test_legacy_train_step_runtime_keeps_optimizer_in_pinned_host_across_steps(self):
    out_dir = self.create_tempdir().full_path

    def _run_legacy_steps(offload_opt: bool):
      cfg = _tiny_m3_config(
          out_dir,
          dataset_type="synthetic",
          optimizer_memory_host_offload=str(offload_opt).lower(),
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
            data_iterator,
            _,
            _,
            _,
            train_state,
        ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
        jit_model, state = nnx.split(train_state)
        p_train_step, _ = train_utils.jit_train_and_eval_step(
            cfg,
            jit_model,
            mesh,
            state,
            state_mesh_shardings,
            legacy_train.train_step,
        )
        batch = next(data_iterator)
        init_opt_kinds = {
            x.sharding.memory_kind
            for x in jax.tree.leaves(state.optimizer)
            if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
        }
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
      losses = [float(metrics0["scalar"]["learning/loss"]), float(metrics1["scalar"]["learning/loss"])]
      return init_opt_kinds, step0_opt_kinds, step1_opt_kinds, step0_param_kinds, losses

    # Negative control: optimizer_memory_host_offload=False keeps optimizer state on device across steps.
    init_dev, step0_dev, step1_dev, param_dev, losses_dev = _run_legacy_steps(offload_opt=False)
    init_host, step0_host, step1_host, param_host, losses_host = _run_legacy_steps(offload_opt=True)

    self.assertEqual(init_dev, {"device"})
    self.assertEqual(step0_dev, {"device"})
    self.assertEqual(step1_dev, {"device"})
    self.assertEqual(param_dev, {"device"})

    # Even though train.py:673 device_puts opt_state to device inside train_step,
    # get_functional_train_with_signature binds out_shardings=(state_mesh_shardings, None),
    # returning updated AdamW moments in pinned_host after step 0 and step 1.
    self.assertEqual(init_host, {"pinned_host"})
    self.assertEqual(step0_host, {"pinned_host"})
    self.assertEqual(step1_host, {"pinned_host"})
    self.assertEqual(param_host, {"device"})
    np.testing.assert_allclose(losses_host, losses_dev, rtol=1e-6, atol=1e-6)

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

  def test_create_model_preserves_qwix_quantization_with_parameter_offload(self):
    """Verifies `model_creation_utils.create_model` keeps Qwix quantization when parameter offload is enabled."""
    toks = jnp.ones((2, 16), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
    for scan_layers in (True, False):
      for remat_pol in ("none", "full"):
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


if __name__ == "__main__":
  absltest.main()

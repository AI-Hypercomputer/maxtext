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

"""Unit tests for startup sharding, synthetic data caching, and host offload init."""

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
from flax.core.spmd import get_logical_axis_rules
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding as maxtext_sharding
from tests.utils.test_helpers import get_test_config_path


class ScaleStartupAndPretrainedTest(unittest.TestCase):
  """Tests Wave 0 startup/sharding fixes on CPU."""

  def test_from_pretrained_applies_logical_axis_rules_in_create_nnx_sharded_model(self):
    """Verifies from_pretrained enters logical_axis_rules around create_nnx_sharded_model."""

    class _DummyModel(nnx.Module):

      def __init__(self, mesh):
        self.mesh = mesh
        self.w = nnx.Param(jnp.ones((4, 4), dtype=jnp.float32))

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    expected_rules = (("embed", "data"), ("mlp", None))
    cfg = SimpleNamespace(
        convert_checkpoint_if_possible=False,
        load_parameters_path="",
        source_checkpoint_layout="orbax",
        logical_axis_rules=expected_rules,
    )
    observed_rules = []

    def _fake_create_sharded(abstract_model, create_fn, mesh=None):
      del abstract_model, mesh
      observed_rules.append(get_logical_axis_rules())
      return create_fn()

    with (
        mock.patch.object(model_creation_utils, "verify_and_sync_scan_layers", side_effect=lambda c: c),
        mock.patch.object(
            model_creation_utils,
            "create_nnx_abstract_model",
            return_value=(lambda: _DummyModel(mesh), nnx.eval_shape(lambda: _DummyModel(mesh))),
        ),
        mock.patch.object(
            model_creation_utils.maxtext_utils_nnx,
            "create_nnx_sharded_model",
            side_effect=_fake_create_sharded,
        ),
    ):
      _ = model_creation_utils.from_pretrained(cfg, mesh=mesh)

    self.assertEqual(observed_rules, [expected_rules])

  def test_synthetic_data_iterator_caches_sharded_batch_and_respects_eval(self):
    """Verifies SyntheticDataIterator builds its sharded batch once in __init__ across Auto/Explicit."""
    live_devices = list(jax.devices())
    for shard_mode, axis_type in (("auto", AxisType.Auto), ("explicit", AxisType.Explicit)):
      mesh = Mesh(np.array(live_devices), ("data",), axis_types=(axis_type,))
      config = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_sharded_synthetic",
          enable_checkpointing=False,
          dataset_type="synthetic",
          per_device_batch_size=2.0,
          eval_per_device_batch_size=1.0,
          mesh_axes=["data"],
          logical_axis_rules=[["batch", "data"]],
          data_sharding=["data"],
          max_target_length=8,
          packing=True,
          shard_mode=shard_mode,
      )
      train_iter = SyntheticDataIterator(config, mesh, is_training=True)
      with mock.patch.object(train_iter, "data_generator", wraps=train_iter.data_generator) as spy_gen:
        b1 = next(train_iter)
        b2 = next(train_iter)
        spy_gen.assert_not_called()

      self.assertIs(b1, b2)
      expected_sharding = maxtext_sharding.get_input_data_sharding(config, mesh)
      self.assertEqual(b1["inputs"].sharding, expected_sharding)
      self.assertEqual(b1["inputs"].shape, (config.global_batch_size_to_load, config.max_target_length))

      eval_iter = SyntheticDataIterator(config, mesh, is_training=False)
      eval_batch = next(eval_iter)
      expected_eval_sharding = maxtext_sharding.get_input_data_sharding(
          config, mesh, rules=config.logical_axis_rules_for_eval
      )
      self.assertEqual(eval_batch["inputs"].sharding, expected_eval_sharding)
      self.assertEqual(eval_batch["inputs"].shape, (config.global_batch_size_to_load_eval, config.max_target_length))

  def test_synthetic_data_iterator_sharding_invariant_under_unsafe_rbg(self):
    """Verifies pinned threefry2x32 keeps synthetic batches bit-identical under prng_impl='unsafe_rbg'."""
    live_devices = list(jax.devices())
    if len(live_devices) < 4:
      self.skipTest("requires at least 4 CPU devices")
    devices4 = np.array(live_devices[:4])

    prev_prng = jax.config.jax_default_prng_impl
    self.addCleanup(jax.config.update, "jax_default_prng_impl", prev_prng)
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    def _build_arm(mesh, *, shard_mode, mesh_axes, rules, data_sharding, per_dev_bs, cp=1):
      cfg = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_synthetic_threefry_invariance",
          enable_checkpointing=False,
          log_config=False,
          skip_jax_distributed_system=True,
          dataset_type="synthetic",
          per_device_batch_size=per_dev_bs,
          mesh_axes=mesh_axes,
          logical_axis_rules=rules,
          data_sharding=data_sharding,
          ici_fsdp_parallelism=4 // cp,
          ici_context_parallelism=cp,
          context_parallel_load_balance=False,
          max_target_length=16,
          packing=True,
          shard_mode=shard_mode,
      )
      it = SyntheticDataIterator(cfg, mesh, is_training=True)
      with jax.set_mesh(mesh):
        lowered = it.data_generator.lower(cfg, True)
        stablehlo_text = lowered.as_text()
        hlo_text = lowered.compile().as_text()
      return next(it), stablehlo_text, hlo_text

    fsdp_rules = [["activation_embed_and_logits_batch", "fsdp"], ["activation_norm_length", None]]
    cp_rules = [["activation_embed_and_logits_batch", "fsdp"], ["activation_norm_length", "context"]]

    b_auto, shlo_auto, hlo_auto = _build_arm(
        Mesh(devices4, ("fsdp",), axis_types=(AxisType.Auto,)),
        shard_mode="auto",
        mesh_axes=["fsdp"],
        rules=fsdp_rules,
        data_sharding=["fsdp"],
        per_dev_bs=2.0,
    )
    b_explicit, shlo_exp, hlo_exp = _build_arm(
        Mesh(devices4, ("fsdp",), axis_types=(AxisType.Explicit,)),
        shard_mode="explicit",
        mesh_axes=["fsdp"],
        rules=fsdp_rules,
        data_sharding=["fsdp"],
        per_dev_bs=2.0,
    )
    b_cp, shlo_cp, hlo_cp = _build_arm(
        Mesh(devices4.reshape(2, 2), ("fsdp", "context"), axis_types=(AxisType.Auto, AxisType.Auto)),
        shard_mode="auto",
        mesh_axes=["fsdp", "context"],
        rules=cp_rules,
        data_sharding=[["fsdp", "context"]],
        per_dev_bs=2.0,
        cp=2,
    )

    self.assertEqual(b_auto["inputs"].sharding.spec, P("fsdp", None))
    self.assertEqual(b_explicit["inputs"].sharding.spec, P("fsdp", None))
    self.assertEqual(b_cp["inputs"].sharding.spec, P("fsdp", "context"))
    self.assertEqual(b_auto["inputs"].shape, (8, 16))
    self.assertEqual(b_cp["inputs"].shape, (8, 16))

    for other in (b_explicit, b_cp):
      for key in ("inputs", "targets", "inputs_segmentation", "targets_segmentation"):
        np.testing.assert_array_equal(np.asarray(jax.device_get(b_auto[key])), np.asarray(jax.device_get(other[key])))

    for text in (shlo_auto, hlo_auto, shlo_exp, hlo_exp, shlo_cp, hlo_cp):
      self.assertNotIn("rng-bit-generator", text)
      self.assertNotIn("rng_bit_generator", text)

  def test_setup_initial_state_two_stage_host_offload_init(self):
    """Verifies setup_initial_state initializes on device before transferring optimizer to pinned_host."""
    num_devices = len(jax.devices())

    class _DummyTrainState(nnx.Module):

      def __init__(self):
        self.model = nnx.Param(jnp.ones((num_devices,), dtype=jnp.float32))
        self.optimizer = nnx.Variable(jnp.zeros((num_devices,), dtype=jnp.float32))

    mesh = Mesh(np.array(jax.devices()), ("data",))
    dev_sharding = NamedSharding(mesh, P("data"), memory_kind="device")
    host_sharding = NamedSharding(mesh, P("data"), memory_kind="pinned_host")

    abstract_state = nnx.eval_shape(_DummyTrainState)
    state_mesh_shardings = nnx.state(_DummyTrainState())
    state_mesh_shardings["model"].set_value(dev_sharding)
    state_mesh_shardings["optimizer"].set_value(host_sharding)
    state_mesh_annotations = nnx.state(_DummyTrainState())

    cfg = SimpleNamespace(
        logical_axis_rules=(("batch", "data"),),
        fp32_master_storage="f32",
        load_parameters_path="",
        load_full_state_path="",
        checkpoint_storage_concurrent_gb=1,
        enable_single_replica_ckpt_restoring=False,
        dataset_type="synthetic",
        checkpoint_storage_use_ocdbt=True,
        checkpoint_storage_use_zarr3=True,
        enable_orbax_v1=False,
        checkpoint_conversion_fn=None,
        source_checkpoint_layout="orbax",
        expansion_factor_real_data=False,
        optimizer_memory_host_offload=True,
        parameter_memory_host_offload=False,
        enable_diloco=False,
    )

    jit_calls = []
    real_jit = jax.jit

    def _recording_jit(fun, *args, **kwargs):
      jit_calls.append(kwargs)
      return real_jit(fun, *args, **kwargs)

    with (
        mock.patch.object(
            maxtext_utils,
            "get_abstract_state",
            return_value=(abstract_state, state_mesh_annotations, state_mesh_shardings),
        ),
        mock.patch.object(maxtext_utils.lineage_adapter, "is_native", return_value=False),
        mock.patch.object(maxtext_utils.checkpointing, "load_state_if_possible", return_value=(None, None)),
        mock.patch("maxtext.utils.maxtext_utils.jax.jit", side_effect=_recording_jit),
    ):
      state, *_ = maxtext_utils.setup_initial_state(
          data_iterator=None,
          config=cfg,
          mesh=mesh,
          checkpoint_manager=None,
          init_state_fn=_DummyTrainState,
          is_training=True,
      )

    self.assertEqual(len(jit_calls), 2)
    self.assertEqual(jit_calls[0]["out_shardings"]["model"].get_value(), dev_sharding)
    self.assertEqual(jit_calls[0]["out_shardings"]["optimizer"].get_value(), dev_sharding)
    self.assertEqual(jit_calls[1]["in_shardings"][0]["optimizer"].get_value(), dev_sharding)
    self.assertEqual(jit_calls[1]["out_shardings"]["optimizer"].get_value(), host_sharding)
    self.assertEqual(jit_calls[1]["donate_argnums"], (0,))
    self.assertEqual(state["model"].get_value().sharding.memory_kind, "device")
    self.assertEqual(state["optimizer"].get_value().sharding.memory_kind, "pinned_host")


if __name__ == "__main__":
  unittest.main()

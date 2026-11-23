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

"""Tests for synthetic data sharding."""

import os
import subprocess
import sys
import types
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import numpy as np

if __name__ == "__main__":
  _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
  _src_dir = os.path.join(_repo_root, "src")
  if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
  if _src_dir not in sys.path:
    sys.path.insert(1, _src_dir)

from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.utils import elastic_utils
from maxtext.utils import sharding as maxtext_sharding
from tests.utils.test_helpers import get_test_config_path


class SyntheticDataShardingTest(parameterized.TestCase):

  @parameterized.product(
      mesh_shape=[(1, 1), (2, 1), (2, 2)],
      shard_mode=["auto", "explicit"],
  )
  def test_synthetic_data_sharding(self, mesh_shape, shard_mode):
    devices = jax.devices()
    num_devices = len(devices)

    target_devices = mesh_shape[0] * mesh_shape[1]

    if num_devices < target_devices:
      self.skipTest(f"Not enough devices. Required: {target_devices}, Available: {num_devices}")

    mesh_devices = devices[:target_devices]
    devices_array = mesh_utils.create_device_mesh(mesh_shape, mesh_devices)
    mesh = Mesh(devices_array, ["data", "fsdp"])

    # Initialize config
    init_kwargs = {
        "per_device_batch_size": 2.0,
        "num_target_devices": len(mesh_devices),
        "run_name": "test",
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "model_name": "llama3.1-8b",
        "max_target_length": 16,
        "shard_mode": shard_mode,
    }
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **init_kwargs,
    )

    iterator = SyntheticDataIterator(config, mesh)
    batch = next(iterator)

    self.assertIn("inputs", batch)
    inputs = batch["inputs"]

    self.assertEqual(inputs.shape, (config.global_batch_size_to_load, config.max_target_length))

    # Expected sharding: mapping of logical axes to physical mesh
    expected_sharding = maxtext_sharding.get_input_data_sharding(config, mesh)
    self.assertEqual(inputs.sharding, expected_sharding)

  def test_constant_padded_masks_train_tail_and_uses_unpadded_eval_batch(self):
    live_devices = list(jax.devices())
    num_live = len(live_devices)
    if num_live < 2:
      if not sys.executable:
        self.skipTest("Needs >= 2 devices.")
      self.assertNotEqual(os.environ.get("_MAXTEXT_CONSTANT_PADDED_SUBPROC"), "1")
      repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
      src_dir = os.path.join(repo_root, "src")
      cur_pp = os.environ.get("PYTHONPATH", "")
      env = {
          **os.environ,
          "_MAXTEXT_CONSTANT_PADDED_SUBPROC": "1",
          "JAX_PLATFORMS": "cpu",
          "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip(),
          "PYTHONPATH": f"{repo_root}:{src_dir}:{cur_pp}".rstrip(":"),
      }
      cmd = [sys.executable, os.path.abspath(__file__), f"{type(self).__name__}.{self._testMethodName}"]
      res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
      self.assertEqual(res.returncode, 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
      return

    all_devices = live_devices + [mock.MagicMock(slice_index=99)]

    with (
        mock.patch("jax.devices", return_value=all_devices),
        mock.patch.object(elastic_utils, "ensure_elastic_manager_initialized"),
        mock.patch.object(elastic_utils, "live_devices", return_value=live_devices),
    ):
      config = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_synthetic_constant_padded",
          enable_checkpointing=False,
          enable_single_controller=True,
          elastic_enabled=True,
          elastic_global_batch_size_mode="constant_padded",
          per_device_batch_size=1.0,
          eval_per_device_batch_size=1.0,
          eval_interval=5,
          dataset_type="synthetic",
          packing=False,
          mesh_axes=["data"],
          logical_axis_rules=[["batch", "data"]],
          data_sharding=["data"],
          max_target_length=8,
      )

    mesh = Mesh(mesh_utils.create_device_mesh((num_live,), live_devices), config.mesh_axes)
    train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config, mesh)
    self.assertIsNotNone(eval_iter)

    train_batch = next(train_iter)
    self.assertEqual(train_batch["inputs"].shape, (config.global_batch_size_to_load, 8))
    in_one = jnp.all(train_batch["inputs_segmentation"] == 1, axis=-1)
    in_zero = jnp.all(train_batch["inputs_segmentation"] == 0, axis=-1)
    self.assertEqual(int(jnp.sum(in_one)), config.logical_global_batch_size_to_load)
    self.assertEqual(int(jnp.sum(in_zero)), config.global_batch_size_to_load - config.logical_global_batch_size_to_load)
    self.assertGreater(int(jnp.sum(in_zero)), 0)
    self.assertTrue(bool(jnp.array_equal(train_batch["inputs_segmentation"], train_batch["targets_segmentation"])))

    eval_batch = next(eval_iter)
    self.assertEqual(eval_batch["inputs"].shape, (config.global_batch_size_to_load_eval, 8))
    self.assertTrue(bool(jnp.all(eval_batch["inputs_segmentation"] == 1)))
    self.assertTrue(bool(jnp.all(eval_batch["targets_segmentation"] == 1)))


class SyntheticDataPaddedRowsTest(parameterized.TestCase):
  """The logical rows of a padded synthetic batch match the rows of the unpadded full-job batch."""

  def _data(self, batch_size, logical_batch_size=None, packing=True, num_replicas=1, replica_index=0):
    """The (tokens, positions, segmentation) rows a `SyntheticDataIterator` builds for one replica."""
    config = types.SimpleNamespace(
        global_batch_size_to_load=batch_size,
        global_batch_size_to_load_eval=batch_size,
        logical_global_batch_size_to_load=logical_batch_size,
        logical_axis_rules_for_eval=None,
        max_target_length=16,
        max_segments_per_seq=4,
        vocab_size=128,
        packing=packing,
        num_data_replicas_per_process=num_replicas,
        data_replica_index=replica_index,
        enable_diloco=False,
    )
    batch = SyntheticDataIterator.raw_generate_synthetic_data(config, is_training=True)
    return batch["inputs"], batch["inputs_position"], batch["inputs_segmentation"]

  @parameterized.product(
      packing=[True, False], replica=[(1, 0), (2, 0), (2, 1)], prng_impl=["unsafe_rbg", "threefry2x32"]
  )
  def test_logical_rows_match_unpadded_batch(self, packing, replica, prng_impl):
    # `base.yml` defaults to `prng_impl: unsafe_rbg`; on TPU a smaller draw is not a row-prefix of a larger one.
    self.addCleanup(jax.config.update, "jax_default_prng_impl", jax.config.jax_default_prng_impl)
    jax.config.update("jax_default_prng_impl", prng_impl)
    num_replicas, replica_index = replica
    full = self._data(4, packing=packing, num_replicas=num_replicas, replica_index=replica_index)
    padded = self._data(6, 4, packing=packing, num_replicas=num_replicas, replica_index=replica_index)
    for full_part, padded_part in zip(full, padded, strict=True):
      self.assertTrue(bool(jnp.array_equal(padded_part[:4], full_part)))
    tokens, _, segmentation = padded
    self.assertTrue(bool(jnp.all(segmentation[4:] == 0)))
    self.assertTrue(bool(jnp.all(tokens[4:] == 0)))
    if packing:
      self.assertGreater(int(jnp.max(full[2])), 1)  # Packed: several segments per row.

  @parameterized.parameters((1, 4), (2, 8))
  def test_packed_segment_ids_are_generated_for_the_logical_rows_only(self, num_replicas, logical_rows):
    # On CPU a smaller draw is a row-prefix of a larger one for both PRNGs, so check the generated shape directly.
    with mock.patch.object(
        synthetic_data_processing,
        "_make_packed_segment_ids",
        wraps=synthetic_data_processing._make_packed_segment_ids,  # pylint: disable=protected-access
    ) as make_segment_ids:
      self._data(6, 4, num_replicas=num_replicas)
    make_segment_ids.assert_called_once_with(logical_rows, 16, 4)


class DeepSeek41SyntheticDataTest(absltest.TestCase):
  """Unit tests for DeepSeek-V4.1 multimodal synthetic data."""

  def test_non_ds41_configs_bit_identical_to_base(self):
    """Non-DS4.1 configs produce bit-identical batches and shardings to the base generator."""
    devices = list(jax.devices())
    mesh = Mesh(np.array(devices), ["data"])
    for packing in (False, True):
      for logical_bs in (None, 2):
        config = pyconfig.initialize(
            [sys.argv[0], get_test_config_path()],
            run_name="test_non_ds41_bit_identical",
            enable_checkpointing=False,
            per_device_batch_size=float(max(4 // len(devices), 1)),
            num_target_devices=len(devices),
            dataset_type="synthetic",
            packing=packing,
            max_segments_per_seq=3,
            mesh_axes=["data"],
            logical_axis_rules=[["batch", "data"]],
            data_sharding=["data"],
            max_target_length=16,
            vocab_size=128,
        )
        if logical_bs is not None:
          object.__setattr__(config, "logical_global_batch_size_to_load", logical_bs)
        iterator = SyntheticDataIterator(config, mesh, is_training=True)
        batch = next(iterator)
        self.assertNotIn("images", batch)

        batch_size = config.global_batch_size_to_load
        eff_logical = batch_size if logical_bs is None else min(logical_bs, batch_size)
        ref_tokens = jax.random.randint(
            jax.random.key(0, impl="threefry2x32"),
            (eff_logical, config.max_target_length + 1),
            0,
            config.vocab_size,
            dtype=jnp.int32,
        )
        ref_tokens = synthetic_data_processing._pad_replica_blocks(ref_tokens, 1, batch_size)  # pylint: disable=protected-access
        ref_pos = jnp.broadcast_to(
            jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1),
            (batch_size, config.max_target_length + 1),
        )
        if packing:
          ref_seg = synthetic_data_processing._make_packed_segment_ids(  # pylint: disable=protected-access
              eff_logical, config.max_target_length, config.max_segments_per_seq
          )
          ref_seg = synthetic_data_processing._pad_replica_blocks(ref_seg, 1, batch_size)  # pylint: disable=protected-access
        else:
          ref_seg = jnp.ones((batch_size, config.max_target_length), dtype=jnp.int32)
        if eff_logical < batch_size:
          ref_seg = ref_seg.at[eff_logical:].set(0)

        self.assertTrue(bool(jnp.array_equal(batch["inputs"], ref_tokens[:, :-1])))
        self.assertTrue(bool(jnp.array_equal(batch["targets"], ref_tokens[:, 1:])))
        self.assertTrue(bool(jnp.array_equal(batch["inputs_position"], ref_pos[:, :-1])))
        self.assertTrue(bool(jnp.array_equal(batch["targets_position"], ref_pos[:, 1:])))
        self.assertTrue(bool(jnp.array_equal(batch["inputs_segmentation"], ref_seg)))
        self.assertTrue(bool(jnp.array_equal(batch["targets_segmentation"], ref_seg)))
        expected_shd = maxtext_sharding.get_input_data_sharding(config, mesh)
        self.assertEqual(batch["inputs"].sharding, expected_shd)

  def test_deepseek41_constant_padded_elastic_zero_masks(self):
    """Verify logical_global_batch_size_to_load matches unpadded logical rows and zero-pads tail rows."""
    live_devices = tuple(jax.devices()[:1])
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_ds41_synthetic_padded",
        enable_checkpointing=False,
        per_device_batch_size=4.0,
        dataset_type="synthetic",
        packing=False,
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        max_target_length=400,
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        ds41_vision_grid=[4, 4],
        ds41_vision_downsample_ratio=2,
        max_num_images_per_example=1,
        ds41_image_token_id=100,
        vocab_size=200,
        num_channels_for_vit=3,
        patch_size_for_vit=16,
    )
    g_load = config.global_batch_size_to_load
    l_load = g_load // 2

    mesh = Mesh(mesh_utils.create_device_mesh((1,), live_devices), config.mesh_axes)
    cfg_unpadded = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_ds41_synthetic_unpadded",
        enable_checkpointing=False,
        per_device_batch_size=2.0,
        dataset_type="synthetic",
        packing=False,
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        max_target_length=400,
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        ds41_vision_grid=[4, 4],
        ds41_vision_downsample_ratio=2,
        max_num_images_per_example=1,
        ds41_image_token_id=100,
        vocab_size=200,
        num_channels_for_vit=3,
        patch_size_for_vit=16,
    )
    unpadded_batch = next(SyntheticDataIterator(cfg_unpadded, mesh, is_training=True))

    object.__setattr__(config, "logical_global_batch_size_to_load", l_load)
    iterator = SyntheticDataIterator(config, mesh, is_training=True)
    train_batch = next(iterator)

    self.assertGreater(g_load, 0)
    self.assertGreater(l_load, 0)
    for key in ("inputs", "targets", "inputs_segmentation", "targets_segmentation", "images"):
      self.assertTrue(bool(jnp.array_equal(train_batch[key][:l_load], unpadded_batch[key])))
      self.assertTrue(bool(jnp.all(train_batch[key][l_load:] == 0)), f"Expected zero padding on {key}")
    self.assertEqual(int(jnp.sum(train_batch["inputs_segmentation"][:l_load])), l_load * config.max_target_length)

  def test_ds41_synthetic_eval_batch_scalar_pspec_and_diloco(self):
    """Verifies DS4.1 synthetic eval batch size, scalar PartitionSpec, DiLoCo 5D, and compound sharding."""
    devices = list(jax.devices())
    mesh_1d = Mesh(np.array(devices), ["data"])

    cfg_eval = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_ds41_synthetic_eval_and_scalar",
        enable_checkpointing=False,
        per_device_batch_size=4.0,
        eval_per_device_batch_size=8.0,
        dataset_type="synthetic",
        max_target_length=24,
        mesh_axes=["data"],
        logical_axis_rules=[["batch", None]],
        data_sharding=[None],
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        ds41_vision_grid=[4, 4],
        ds41_vision_downsample_ratio=2,
        max_num_images_per_example=1,
        ds41_image_token_id=100,
        vocab_size=200,
        num_channels_for_vit=3,
        patch_size_for_vit=16,
    )
    expected_train_bs = int(cfg_eval.global_batch_size_to_train_on)
    expected_eval_bs = int(cfg_eval.global_batch_size_to_eval_on)
    self.assertNotEqual(expected_eval_bs, expected_train_bs)
    scalar_shd = jax.sharding.NamedSharding(mesh_1d, jax.sharding.PartitionSpec())
    with mock.patch.object(maxtext_sharding, "get_input_data_sharding", return_value=scalar_shd):
      raw_eval_it = SyntheticDataIterator(cfg_eval, mesh_1d, is_training=False)
      eval_batch = next(raw_eval_it)
    self.assertEqual(eval_batch["inputs"].shape, (expected_eval_bs, 24))
    self.assertEqual(eval_batch["images"].shape[0], expected_eval_bs)
    self.assertEqual(eval_batch["images"].sharding.spec, jax.sharding.PartitionSpec())

    if len(devices) >= 2:
      cfg_diloco = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_ds41_synthetic_diloco",
          enable_checkpointing=False,
          enable_diloco=True,
          ici_diloco_parallelism=2,
          ici_data_parallelism=max(len(devices) // 2, 1),
          ici_fsdp_parallelism=1,
          per_device_batch_size=4.0,
          dataset_type="synthetic",
          max_target_length=24,
          use_multimodal=True,
          vision_encoder_block="deepseek41",
          ds41_vision_grid=[4, 4],
          ds41_vision_downsample_ratio=2,
          max_num_images_per_example=1,
          ds41_image_token_id=100,
          vocab_size=200,
          num_channels_for_vit=3,
          patch_size_for_vit=16,
      )
      mesh_diloco = Mesh(
          np.array(devices).reshape((2, max(len(devices) // 2, 1)) + (1,) * (len(cfg_diloco.mesh_axes) - 2)),
          cfg_diloco.mesh_axes,
      )
      raw_diloco_batch = next(SyntheticDataIterator(cfg_diloco, mesh_diloco, is_training=True))
      self.assertEqual(raw_diloco_batch["images"].ndim, 5)
      self.assertEqual(
          raw_diloco_batch["images"].shape[:2],
          (2, int(cfg_diloco.global_batch_size_to_train_on) // 2),
      )

      compound_shd = jax.sharding.NamedSharding(mesh_diloco, jax.sharding.PartitionSpec("diloco", ("data", "fsdp"), None))
      with mock.patch.object(maxtext_sharding, "get_input_data_sharding", return_value=compound_shd):
        raw_compound_batch = next(SyntheticDataIterator(cfg_diloco, mesh_diloco, is_training=True))
      self.assertEqual(raw_compound_batch["images"].ndim, 5)
      self.assertEqual(
          raw_compound_batch["images"].sharding.spec,
          jax.sharding.PartitionSpec("diloco", ("data", "fsdp"), None, None, None),
      )

  def test_ds41_validation_runs_even_when_config_mutated_in_place(self):
    """Validation runs eagerly in __init__ even if HyperParameters is mutated in place after first jit."""
    devices = list(jax.devices()[:1])
    mesh = Mesh(np.array(devices), ["data"])
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_ds41_mutated_cfg",
        enable_checkpointing=False,
        per_device_batch_size=2.0,
        dataset_type="synthetic",
        max_target_length=64,
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        ds41_vision_grid=[7, 5],
        ds41_vision_downsample_ratio=3,
        max_num_images_per_example=1,
        ds41_image_token_id=200,
        vocab_size=256,
        num_channels_for_vit=3,
        patch_size_for_vit=14,
    )
    ok_batch = next(SyntheticDataIterator(cfg, mesh))
    self.assertTrue(bool(jnp.all(ok_batch["inputs"][:, 8:19] == 200)))
    for bad_tok_id in (-1, 0, 256, 300):
      cfg.get_keys()["ds41_image_token_id"] = bad_tok_id
      with self.assertRaisesRegex(ValueError, "ds41_image_token_id"):
        SyntheticDataIterator(cfg, mesh)

  def test_ds41_dynamic_vision_grid_synthetic_batch(self):
    """Verifies ds41_dynamic_vision_grid=True pads synthetic patches to P_max and emits image_masks and image_grids."""
    devices = list(jax.devices()[:1])
    mesh = Mesh(np.array(devices), ["data"])
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_ds41_dynamic_synth",
        enable_checkpointing=False,
        per_device_batch_size=2.0,
        dataset_type="synthetic",
        max_target_length=64,
        use_multimodal=True,
        vision_encoder_block="deepseek41",
        ds41_dynamic_vision_grid=True,
        ds41_vision_grid=[5, 8],
        ds41_vision_downsample_ratio=3,
        ds41_vision_max_n_token=20,
        max_num_images_per_example=2,
        ds41_image_token_id=200,
        vocab_size=256,
        num_channels_for_vit=3,
        patch_size_for_vit=14,
    )
    batch = next(SyntheticDataIterator(cfg, mesh))
    bsz = int(cfg.global_batch_size_to_load)
    p_max = 9 * (20 - 3)
    self.assertEqual(batch["images"].shape, (bsz, 2, p_max, 588))
    self.assertEqual(batch["image_masks"].shape, (bsz, 2))
    self.assertEqual(batch["image_grids"].shape, (bsz, 2, 2))
    np.testing.assert_array_equal(np.asarray(batch["image_masks"]), np.ones((bsz, 2), dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image_grids"]), np.broadcast_to(np.array([5, 8], dtype=np.int32), (bsz, 2, 2))
    )
    np.testing.assert_array_equal(np.asarray(batch["images"])[:, :, 40:, :], 0.0)


class PlaceHolderDataIteratorTest(absltest.TestCase):
  """Tests for the placeholder iterator used by the non loading hosts."""

  def _config(self):
    return types.SimpleNamespace(
        global_batch_size_to_load=16,
        global_batch_size_to_load_eval=8,
        max_target_length=4,
    )

  def test_eval_placeholder_uses_eval_batch_size(self):
    data = next(PlaceHolderDataIterator.get_place_holder_synthetic_data(self._config(), is_training=False))
    self.assertEqual(data["inputs"].shape, (8 // jax.process_count(), 4))

  def test_train_placeholder_uses_train_batch_size(self):
    data = next(PlaceHolderDataIterator.get_place_holder_synthetic_data(self._config()))
    self.assertEqual(data["inputs"].shape, (16 // jax.process_count(), 4))


if __name__ == "__main__":
  absltest.main()

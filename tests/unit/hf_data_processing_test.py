# Copyright 2023–2025 Google LLC
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

"""Tests for Hugging Face data processing."""

import sys
from types import SimpleNamespace
import unittest
import os.path

import jax
import pytest
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path, get_test_base_output_directory


@pytest.mark.cpu_only
class HfDataProcessingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    decoupled = is_decoupled()
    # Note: this test uses gs://max-experiments/ (not gs://runner-maxtext-logs)
    base_output_directory = get_test_base_output_directory(cloud_path="gs://max-experiments/")
    self.config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory=base_output_directory,
        dataset_type="hf",
        hf_path="parquet",
        hf_data_dir="",
        hf_train_files=(
            os.path.join(
                "tests",
                "assets",
                "local_datasets",
                "c4_en_dataset_minimal",
                "hf",
                "c4",
                "c4-train-00000-of-01637.parquet",
            )
            if decoupled
            else "gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet"
        ),
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"),
        enable_checkpointing=False,
    )
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.process_indices = input_pipeline_interface.get_process_loading_real_data(
        self.config.data_sharding,
        self.config.global_batch_size_to_load,
        self.config.global_batch_size_to_train_on,
        self.config.max_target_length,
        self.mesh,
    )

  @property
  def train_iter(self):
    # pylint: disable=protected-access
    if not hasattr(self.__class__, "_cached_train_iter"):
      self.__class__._cached_train_iter = hf_data_processing.make_hf_train_iterator(
          self.config, self.mesh, self.process_indices
      )
    return self.__class__._cached_train_iter

  def test_train_ds(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    batch = next(self.train_iter)
    self.assertEqual(
        {k: list(v.shape) for k, v in batch.items()},
        {
            "inputs": expected_shape,
            "inputs_position": expected_shape,
            "inputs_segmentation": expected_shape,
            "targets": expected_shape,
            "targets_position": expected_shape,
            "targets_segmentation": expected_shape,
        },
    )

  def test_batch_determinism(self):
    batch1 = next(self.train_iter)
    train_iter = hf_data_processing.make_hf_train_iterator(self.config, self.mesh, self.process_indices)
    batch2 = next(train_iter)
    self.assertTrue((batch1["inputs"] == batch2["inputs"]).all())
    self.assertTrue((batch1["targets"] == batch2["targets"]).all())
    self.assertTrue((batch1["inputs_segmentation"] == batch2["inputs_segmentation"]).all())
    self.assertTrue((batch1["targets_segmentation"] == batch2["targets_segmentation"]).all())
    self.assertTrue((batch1["inputs_position"] == batch2["inputs_position"]).all())
    self.assertTrue((batch1["targets_position"] == batch2["targets_position"]).all())

  def test_for_loop_repeatable(self):
    def get_first_batch(iterator):
      batch = None
      for batch in iterator:
        break
      return batch

    train_batch1 = get_first_batch(self.train_iter)
    train_batch2 = get_first_batch(self.train_iter)
    self.assertTrue((train_batch1["inputs"] == train_batch2["inputs"]).all())  # pytype: disable=unsupported-operands
    self.assertTrue((train_batch1["targets"] == train_batch2["targets"]).all())  # pytype: disable=unsupported-operands


@pytest.mark.cpu_only
class TrainingObjectiveTransformTest(unittest.TestCase):
  """Tests objective selection at the HF pipeline boundary."""

  def test_default_objective_keeps_shift_data(self):
    transform = hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
        SimpleNamespace(),
        shift=True,
        use_dpo=False,
        use_sft=False,
        completion_only=False,
        packing=False,
        pad_id=0,
        bos_token_id=1,
    )

    self.assertIsInstance(transform, input_pipeline_utils.ShiftData)
    self.assertEqual(transform.ignored_ids, [0, 1])

  def test_explicit_block_diffusion_objective_replaces_shift(self):
    config = SimpleNamespace(
        training_objective="block_diffusion",
        block_diffusion_block_size=4,
        block_diffusion_mask_id=99,
        block_diffusion_min_noise=0.05,
        block_diffusion_canvas_policy="seed_and_mask",
        block_diffusion_logit_alignment="shifted",
    )

    transform = hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
        config,
        shift=True,
        use_dpo=False,
        use_sft=True,
        completion_only=True,
        packing=False,
        pad_id=0,
        bos_token_id=1,
    )

    self.assertIsInstance(transform, input_pipeline_utils.BlockDiffusionCorruption)
    self.assertEqual(transform.block_size, 4)
    self.assertEqual(transform.mask_id, 99)
    self.assertEqual(transform.min_noise, 0.05)
    self.assertTrue(transform.completion_only)
    self.assertTrue(transform.seed_first_token)
    self.assertTrue(transform.include_seed_in_loss)

  def test_block_diffusion_rejects_packing(self):
    config = SimpleNamespace(
        training_objective="block_diffusion",
        block_diffusion_block_size=4,
        block_diffusion_mask_id=99,
        block_diffusion_min_noise=0.05,
        block_diffusion_canvas_policy="all_masked",
    )

    with self.assertRaisesRegex(ValueError, "packing=False"):
      hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
          config,
          shift=True,
          use_dpo=False,
          use_sft=False,
          completion_only=False,
          packing=True,
          pad_id=0,
          bos_token_id=1,
      )


if __name__ == "__main__":
  unittest.main()

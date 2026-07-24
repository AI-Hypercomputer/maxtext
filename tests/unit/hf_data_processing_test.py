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
import unittest
import os.path
from types import SimpleNamespace
from unittest import mock

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path, get_test_base_output_directory


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
class HfEvalIteratorBranchTest(unittest.TestCase):
  """Lightweight branch tests for make_hf_eval_iterator."""

  def test_multimodal_sft_eval_uses_vision_pipeline(self):
    config = SimpleNamespace(
        hf_path="parquet",
        hf_name=None,
        hf_data_dir="",
        hf_eval_files="eval.parquet",
        hf_eval_split="train",
        hf_access_token="",
        use_sft=True,
        use_multimodal=True,
        eval_data_columns=("messages",),
        eval_image_column="image",
        global_batch_size_to_load_eval=4,
    )
    fake_dataset = object()
    fake_datasets_module = SimpleNamespace(load_dataset=mock.Mock(return_value=fake_dataset))
    sentinel = object()

    with (
        mock.patch.dict(sys.modules, {"datasets": fake_datasets_module}),
        mock.patch.object(
            hf_data_processing,
            "vision_sft_preprocessing_pipeline",
            return_value=sentinel,
        ) as vision_pipeline,
    ):
      result = hf_data_processing.make_hf_eval_iterator(config, global_mesh="mesh", process_indices_eval=[0])

    self.assertIs(result, sentinel)
    fake_datasets_module.load_dataset.assert_called_once_with(
        "parquet",
        name=None,
        data_dir="",
        data_files="eval.parquet",
        split="train",
        streaming=True,
        token="",
    )
    vision_pipeline.assert_called_once_with(
        dataset=fake_dataset,
        config=config,
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh="mesh",
        text_columns=("messages",),
        image_column="image",
        global_batch_size=4,
    )


if __name__ == "__main__":
  unittest.main()

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

"""Unit tests for multimodal SFT grain input pipeline."""

import unittest
from PIL import Image
import grain.python as grain

from maxtext.configs import pyconfig
from maxtext.input_pipeline.grain_data_processing import (
    vision_sft_preprocessing_pipeline,
    _get_pipeline_fn,
)


class MultimodalSftGrainTest(unittest.TestCase):
  """Tests for the Grain multimodal SFT pipeline."""

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-chartqa.yml",
            "model_name=gemma3-4b",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "max_prefill_predict_length=512",
            "tokenize_train_data=True",
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
            "dataset_type=grain",
            "grain_worker_count=0",
        ]
    )

  def test_get_pipeline_fn_routes_to_vision_sft(self):
    """Verifies that _get_pipeline_fn routes to vision_sft_preprocessing_pipeline."""
    fn = _get_pipeline_fn(self.config)
    self.assertEqual(fn, vision_sft_preprocessing_pipeline)

  def test_vision_sft_pipeline_execution(self):
    """Tests executing the Grain vision SFT pipeline with dummy samples."""
    dummy_image = Image.new("RGB", (100, 100), color="red")
    raw_data = [
        {
            "query": f"What is the chart {i} about?",
            "label": [f"Sales in 202{i}"],
            "image": dummy_image,
        }
        for i in range(8)
    ]
    ds = grain.MapDataset.source(raw_data)
    processed_ds = vision_sft_preprocessing_pipeline(
        dataset=ds,
        config=self.config,
        data_columns=["query", "label"],
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batches = list(processed_ds)
    self.assertGreater(len(batches), 0)
    first_batch = batches[0]

    for expected_key in [
        "inputs",
        "targets",
        "images",
        "inputs_position",
        "targets_position",
        "inputs_segmentation",
        "targets_segmentation",
    ]:
      self.assertIn(expected_key, first_batch)

    self.assertEqual(first_batch["inputs"].shape[-1], self.config.max_target_length)
    self.assertEqual(first_batch["targets"].shape[-1], self.config.max_target_length)

  def test_elastic_iterator_unsupported_error(self):
    """Verifies that enabling grain_use_elastic_iterator raises a ValueError."""
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-chartqa.yml",
            "model_name=gemma3-4b",
            "grain_file_type=arrayrecord",
            "grain_train_files=dummy",
            "grain_use_elastic_iterator=True",
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
        ]
    )
    raw_data = [{"query": "q", "label": ["l"], "image": Image.new("RGB", (10, 10))}]
    ds = grain.MapDataset.source(raw_data)
    with self.assertRaises(ValueError):
      vision_sft_preprocessing_pipeline(
          dataset=ds,
          config=config,
          data_columns=["query", "label"],
          tokenize=True,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )

  def test_slidevqa_config_initialization(self):
    """Verifies that SlideVQA config initializes cleanly with dataset_type=grain."""
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-slidevqa.yml",
            "model_name=gemma3-4b",
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
        ]
    )
    self.assertEqual(config.dataset_type, "grain")
    self.assertEqual(config.grain_file_type, "parquet")
    fn = _get_pipeline_fn(config)
    self.assertEqual(fn, vision_sft_preprocessing_pipeline)


if __name__ == "__main__":
  unittest.main()

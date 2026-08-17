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

"""Unit tests for Grain SFT data processing pipelines."""

import unittest
from datasets import Dataset
import grain.python as grain
import jax
from jax.sharding import Mesh
import numpy as np
from PIL import Image

from maxtext.configs import pyconfig
from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import grain_tokenizer
from maxtext.input_pipeline import hf_data_processing


class GrainSftTest(unittest.TestCase):
  """Tests for Grain SFT pipelines and equivalence with Hugging Face pipelines."""

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
    fn = grain_data_processing._get_pipeline_fn(self.config)  # pylint: disable=protected-access
    self.assertEqual(fn, grain_data_processing.vision_sft_preprocessing_pipeline)

  def test_tokenizer_no_bos_eos_in_vision_sft(self):
    """Verifies that get_tokenizer_and_pad_id with add_bos=False and add_eos=False suppresses BOS/EOS."""
    tok, _ = data_processing_utils.get_tokenizer_and_pad_id(self.config, add_bos=False, add_eos=False)
    elem = {"text": "hello world"}
    transform = grain_tokenizer.TokenizeAndTrim("text", self.config.max_target_length, tok)
    res = transform.map(elem)
    # Ensure BOS (token 2 in Gemma) is not added at the beginning
    self.assertNotEqual(res["text"][0], tok.bos_id)

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
      grain_data_processing.vision_sft_preprocessing_pipeline(
          dataset=ds,
          config=config,
          data_columns=["query", "label"],
          image_column="image",
          tokenize=True,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )

  def test_hf_and_grain_batch_exact_match(self):
    """Verifies that HF and Grain vision SFT pipelines produce identical batch data."""
    dummy_image = Image.new("RGB", (100, 100), color="green")
    samples = [
        {
            "query": f"What does chart {i} represent?",
            "label": [f"Result is {i * 10}"],
            "image": dummy_image,
        }
        for i in range(8)
    ]

    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-chartqa.yml",
            "model_name=qwen3-vl-2b",
            "scan_layers=False",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "max_prefill_predict_length=512",
            "tokenize_train_data=True",
            "tokenizer_type=huggingface",
            "tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer",
            "enable_data_shuffling=False",
            "grain_worker_count=0",
        ]
    )

    # Process using Hugging Face pipeline
    devices = np.array(jax.devices()[:1]).reshape(
        1,
    )
    mesh = Mesh(devices, ("data",))

    hf_ds = Dataset.from_list(samples).to_iterable_dataset()
    hf_processed = hf_data_processing.vision_sft_preprocessing_pipeline(
        dataset=hf_ds,
        config=config,
        dataloading_host_index=0,
        dataloading_host_count=1,
        global_mesh=mesh,
        text_columns=["query", "label"],
        image_column="image",
        global_batch_size=config.global_batch_size_to_load,
    )
    hf_batch = next(iter(hf_processed))

    # Process using Grain pipeline
    grain_ds = grain.MapDataset.source(samples)
    grain_processed = grain_data_processing.vision_sft_preprocessing_pipeline(
        dataset=grain_ds,
        config=config,
        data_columns=["query", "label"],
        image_column="image",
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )
    grain_batch = next(iter(grain_processed))

    # Verify all output arrays match exactly
    for key in [
        "inputs",
        "targets",
        "images",
        "inputs_position",
        "targets_position",
        "inputs_segmentation",
        "targets_segmentation",
    ]:
      self.assertIn(key, hf_batch, msg=f"Key {key} missing from HF batch")
      self.assertIn(key, grain_batch, msg=f"Key {key} missing from Grain batch")
      np.testing.assert_array_equal(
          grain_batch[key],
          hf_batch[key],
          err_msg=f"Mismatch between HF and Grain pipelines for key '{key}'",
      )

  def test_slidevqa_multi_image_pipeline_execution(self):
    """Tests executing Grain vision SFT pipeline with SlideVQA multi-image format."""
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-slidevqa.yml",
            "model_name=gemma3-4b",
            "per_device_batch_size=1",
            "max_target_length=8192",
            "max_prefill_predict_length=512",
            "tokenize_train_data=True",
            "tokenizer_path=src/maxtext/assets/tokenizers/tokenizer.gemma3",
            "dataset_type=grain",
            "grain_worker_count=0",
        ]
    )
    dummy_image = Image.new("RGB", (50, 50), color="blue")
    raw_data = [
        {
            "question": f"Question {i}?",
            "answer": [f"Answer {i}"],
            **{f"page_{p}": dummy_image for p in range(1, 21)},
        }
        for i in range(4)
    ]
    ds = grain.MapDataset.source(raw_data)
    processed_ds = grain_data_processing.vision_sft_preprocessing_pipeline(
        dataset=ds,
        config=config,
        data_columns=list(config.train_data_columns),
        image_column=config.train_image_column,
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )

    batches = list(processed_ds)
    self.assertGreater(len(batches), 0)
    first_batch = batches[0]
    self.assertIn("images", first_batch)
    self.assertIn("inputs", first_batch)
    self.assertIn("targets", first_batch)

  def test_qwen3_vl_2b_mrope_positions(self):
    """Verifies that Qwen3-VL-2B outputs 3D position IDs (batch, seq, 3) under Grain pipeline."""
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/post_train/sft-vision-chartqa.yml",
            "model_name=qwen3-vl-2b",
            "scan_layers=False",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "max_prefill_predict_length=512",
            "tokenize_train_data=True",
            "tokenizer_type=huggingface",
            "tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer",
            "dataset_type=grain",
            "grain_worker_count=0",
        ]
    )
    dummy_image = Image.new("RGB", (100, 100), color="red")
    raw_data = [
        {
            "query": f"What is in chart {i}?",
            "label": [f"Data {i}"],
            "image": dummy_image,
        }
        for i in range(8)
    ]
    ds = grain.MapDataset.source(raw_data)
    processed_ds = grain_data_processing.vision_sft_preprocessing_pipeline(
        dataset=ds,
        config=config,
        data_columns=["query", "label"],
        image_column=config.train_image_column,
        tokenize=True,
        grain_worker_count=0,
        grain_per_worker_buffer_size=1,
    )
    batches = list(processed_ds)
    self.assertGreater(len(batches), 0)
    first_batch = batches[0]
    self.assertIn("inputs_position", first_batch)
    self.assertEqual(
        first_batch["inputs_position"].shape,
        (config.global_batch_size_to_load, config.max_target_length, 3),
    )
    self.assertEqual(
        first_batch["targets_position"].shape,
        (config.global_batch_size_to_load, config.max_target_length),
    )
    self.assertNotIn("inputs_mrope_deltas", first_batch)
    self.assertNotIn("image_grid_thw", first_batch)


if __name__ == "__main__":
  unittest.main()

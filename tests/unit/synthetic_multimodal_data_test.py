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

"""Tests for the synthetic multimodal dataset (`SyntheticMultimodalDataSource`)."""

import sys
from unittest import mock

from absl.testing import absltest
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.multimodal import utils as mm_utils
from tests.utils.test_helpers import get_test_config_path


def _make_config(**overrides):
  """A multimodal config with small synthetic images, overridable per test."""
  kwargs = {
      "run_name": "test_synthetic_multimodal",
      "enable_checkpointing": False,
      "model_name": "gemma3-4b",
      "dataset_type": "synthetic",
      "use_multimodal": True,
      "max_target_length": 64,
      "max_prefill_predict_length": 32,
      "synthetic_image_height_range": [240, 480],
      "synthetic_image_width_range": [320, 640],
      "scan_layers": False,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class SyntheticImageGenerationTest(absltest.TestCase):

  def test_images_follow_the_configured_ranges(self):
    config = _make_config(synthetic_num_images_per_example_range=[0, 3])
    rng = np.random.default_rng(0)
    counts = set()
    for _ in range(50):
      images = synthetic_data_processing.generate_synthetic_images(rng, config)
      counts.add(len(images))
      for image in images:
        height, width, channels = image.shape
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(channels, 3)
        self.assertBetween(height, 240, 480)
        self.assertBetween(width, 320, 640)
    self.assertEqual(counts, {0, 1, 2, 3})

  def test_largest_takes_the_maximum_of_every_range(self):
    config = _make_config(synthetic_num_images_per_example_range=[0, 3])
    images = synthetic_data_processing.generate_synthetic_images(np.random.default_rng(0), config, largest=True)
    self.assertLen(images, 3)
    for image in images:
      self.assertEqual(image.shape, (480, 640, 3))

  def test_images_are_not_constant_and_load_like_dataset_images(self):
    config = _make_config()
    images = synthetic_data_processing.generate_synthetic_images(np.random.default_rng(0), config)
    self.assertLen(images, 1)
    self.assertGreater(float(images[0].std()), 50.0)
    # The image pipelines take dataset images through `convert_to_RGB` + `np.array`; a raw array passes unchanged.
    np.testing.assert_array_equal(np.array(mm_utils.convert_to_RGB(images[0])), images[0])


class SyntheticMultimodalDataSourceTest(absltest.TestCase):

  def test_rows_hold_text_and_media_by_modality(self):
    config = _make_config(synthetic_num_images_per_example_range=[2, 2])
    source = synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=0)
    self.assertEqual(source.modalities, ["image"])
    row = source[7]
    self.assertEqual(set(row), {"prompt", "completion", "image"})
    self.assertLen(row["prompt"].split(), 24)
    # Enough words to fill any sequence once tokenized and truncated; letters only, so no special tokens.
    self.assertLen(row["completion"].split(), config.max_target_length)
    self.assertTrue(row["prompt"].replace(" ", "").isalpha())
    self.assertTrue(row["completion"].replace(" ", "").isalpha())
    self.assertLen(row["image"], 2)
    self.assertEqual(row["image"][0].dtype, np.uint8)

  def test_rows_are_deterministic_in_seed_and_index(self):
    config = _make_config(synthetic_num_images_per_example_range=[0, 2])
    source_a = synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=3)
    source_b = synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=3)
    source_c = synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=4)
    for index in range(4):
      a, b, c = source_a[index], source_b[index], source_c[index]
      self.assertEqual(a["prompt"], b["prompt"])
      self.assertEqual(a["completion"], b["completion"])
      self.assertLen(a["image"], len(b["image"]))
      for image_a, image_b in zip(a["image"], b["image"]):
        np.testing.assert_array_equal(image_a, image_b)
      self.assertNotEqual(a["completion"], c["completion"])
    self.assertNotEqual(source_a[0]["completion"], source_a[1]["completion"])

  def test_largest_row(self):
    config = _make_config(synthetic_num_images_per_example_range=[0, 2])
    source = synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=0)
    row = source.generate(np.random.default_rng(0), largest=True)
    self.assertLen(row["image"], 2)
    self.assertEqual(row["image"][0].shape, (480, 640, 3))

  def test_no_modality_without_images(self):
    config = _make_config(synthetic_num_images_per_example_range=[0, 0])
    self.assertEqual(synthetic_data_processing.synthetic_modalities(config), [])
    with self.assertRaisesRegex(ValueError, "modality"):
      synthetic_data_processing.SyntheticMultimodalDataSource(config, seed=0)
    self.assertEqual(synthetic_data_processing.synthetic_modalities(_make_config(use_multimodal=False)), [])

  def test_invalid_ranges_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "synthetic_image_height_range"):
      _make_config(synthetic_image_height_range=[0, 480])
    with self.assertRaisesRegex(ValueError, "synthetic_num_images_per_example_range"):
      _make_config(synthetic_num_images_per_example_range=[2, 1])


class _DummyQwenTokenizer:
  """Minimal deterministic tokenizer supporting Qwen3-VL special tokens for offline tests."""

  pad_id = 0
  unk_id = 0
  bos_id = 1
  eos_id = 2
  _SPECIAL = {
      "<|vision_start|>": 151652,
      "<|vision_end|>": 151653,
      "<|image_pad|>": 151655,
      "<|video_pad|>": 151656,
      "<|im_start|>": 151644,
      "<|im_end|>": 151645,
  }

  def encode(self, text: str) -> list[int]:
    """Encodes text into deterministic token IDs, preserving Qwen3-VL special tokens."""
    tokens = []
    i = 0
    while i < len(text):
      matched = False
      for sym, tok_id in self._SPECIAL.items():
        if text.startswith(sym, i):
          tokens.append(tok_id)
          i += len(sym)
          matched = True
          break
      if not matched:
        if text[i].isspace():
          i += 1
        else:
          j = i
          while j < len(text) and not text[j].isspace() and text[j] != "<":
            j += 1
          word = text[i:j]
          tokens.append(10 + (sum(ord(c) for c in word) % 1000))
          i = j
    return tokens


class SyntheticMultimodalIteratorTest(absltest.TestCase):

  def _make_qwen_vl_config(self, **overrides):
    """Builds a Qwen3-VL synthetic multimodal test config with optional overrides."""
    kwargs = {
        "model_name": "qwen3-vl-2b",
        "use_multimodal": True,
        "use_sft": True,
        "sft_train_on_completion_only": True,
        "packing": False,
        "dataset_type": "synthetic",
        "per_device_batch_size": 1,
        "max_target_length": 1024,
        "max_prefill_predict_length": 512,
        "synthetic_image_height_range": [224, 512],
        "synthetic_image_width_range": [224, 512],
        "synthetic_num_images_per_example_range": [1, 1],
        "data_shuffle_seed": 42,
        "grain_worker_count": 0,
        "eval_interval": 0,
    }
    kwargs.update(overrides)
    return _make_config(**kwargs)

  def _make_mesh(self, config):
    mesh_shape = (1,) * (len(config.mesh_axes) - 1) + (jax.device_count(),)
    devices_array = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices_array, config.mesh_axes)

  def test_create_data_iterator_qwen3_vl_shapes_and_determinism(self):
    for worker_count in (0, 2):
      with self.subTest(grain_worker_count=worker_count):
        config = self._make_qwen_vl_config(grain_worker_count=worker_count)
        mesh = self._make_mesh(config)
        with mock.patch(
            "maxtext.input_pipeline.tokenizer.build_tokenizer",
            return_value=_DummyQwenTokenizer(),
        ):
          train_iter_a, eval_iter_a = input_pipeline_interface.create_data_iterator(config, mesh)
          train_iter_b, _ = input_pipeline_interface.create_data_iterator(config, mesh)
          self.assertIsNone(eval_iter_a)

          for _ in range(2):
            batch_a = next(train_iter_a)
            batch_b = next(train_iter_b)
            self.assertEqual(set(batch_a.keys()), set(batch_b.keys()))
            self.assertIn("images", batch_a)
            self.assertEqual(batch_a["inputs"].shape, (config.global_batch_size_to_load, config.max_target_length))
            self.assertEqual(
                batch_a["inputs_position"].shape, (config.global_batch_size_to_load, config.max_target_length, 3)
            )
            self.assertEqual(batch_a["images"].shape, (config.global_batch_size_to_load, 3, 2, 768, 768))
            for key in batch_a:
              np.testing.assert_array_equal(np.asarray(batch_a[key]), np.asarray(batch_b[key]))

  def test_different_seed_produces_different_batches(self):
    config_a = self._make_qwen_vl_config(data_shuffle_seed=1)
    config_b = self._make_qwen_vl_config(data_shuffle_seed=2)
    mesh = self._make_mesh(config_a)
    with mock.patch(
        "maxtext.input_pipeline.tokenizer.build_tokenizer",
        return_value=_DummyQwenTokenizer(),
    ):
      iter_a, _ = input_pipeline_interface.create_data_iterator(config_a, mesh)
      iter_b, _ = input_pipeline_interface.create_data_iterator(config_b, mesh)
      batch_a = next(iter_a)
      batch_b = next(iter_b)
      self.assertFalse(np.array_equal(np.asarray(batch_a["images"]), np.asarray(batch_b["images"])))

  def test_zero_images_falls_back_to_text_synthetic_iterator(self):
    config = self._make_qwen_vl_config(synthetic_num_images_per_example_range=[0, 0])
    mesh = self._make_mesh(config)
    train_iter, _ = input_pipeline_interface.create_data_iterator(config, mesh)
    self.assertIsInstance(train_iter, synthetic_data_processing.SyntheticDataIterator)

  def test_eval_iterator_uses_eval_batch_size_and_offset_seed(self):
    config = self._make_qwen_vl_config(
        eval_interval=1,
        per_device_batch_size=1,
        eval_per_device_batch_size=2,
    )
    mesh = self._make_mesh(config)
    with mock.patch(
        "maxtext.input_pipeline.tokenizer.build_tokenizer",
        return_value=_DummyQwenTokenizer(),
    ):
      train_iter, eval_iter_a = input_pipeline_interface.create_data_iterator(config, mesh)
      _, eval_iter_b = input_pipeline_interface.create_data_iterator(config, mesh)
      self.assertIsNotNone(eval_iter_a)

      train_batch = next(train_iter)
      eval_batch_a = next(eval_iter_a)
      eval_batch_b = next(eval_iter_b)

      eval_bs = config.global_batch_size_to_load_eval
      self.assertEqual(eval_batch_a["inputs"].shape, (eval_bs, config.max_target_length))
      self.assertEqual(eval_batch_a["inputs_position"].shape, (eval_bs, config.max_target_length, 3))
      self.assertEqual(eval_batch_a["images"].shape, (eval_bs, 3, 2, 768, 768))
      for key in eval_batch_a:
        np.testing.assert_array_equal(np.asarray(eval_batch_a[key]), np.asarray(eval_batch_b[key]))
      self.assertFalse(
          np.array_equal(
              np.asarray(train_batch["images"][:1]),
              np.asarray(eval_batch_a["images"][:1]),
          )
      )


if __name__ == "__main__":
  absltest.main()

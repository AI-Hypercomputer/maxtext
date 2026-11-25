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

from absl.testing import absltest
import numpy as np

from maxtext.configs import pyconfig
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


if __name__ == "__main__":
  absltest.main()

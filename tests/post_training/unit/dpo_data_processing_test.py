# Copyright 2025–2026 Google LLC
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

"""Unit tests for DPO data preparation."""
import os
import unittest
from datasets import Dataset
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import numpy as np
import pytest
import transformers

from maxtext.configs import pyconfig
from maxtext.input_pipeline import dpo_utils
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR, MAXTEXT_PKG_DIR

pytestmark = [pytest.mark.post_training, pytest.mark.cpu_only]


class TestDPODataFormatting(unittest.TestCase):
  """Tests for DPODataFormatting transform."""

  def setUp(self):
    self.pad_id = 0

  def test_column_remapping(self):
    """Verify that columns are renamed to match Tunix expectations."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=21,
        data_column_names=("input", "chosen", "rejected"),
    )
    sample = {
        "input": np.array([1, 2, 3]),
        "chosen": np.array([4, 5]),
        "rejected": np.array([6, 7, 8]),
    }
    output = prep.map(sample)

    # Check that old keys are removed
    self.assertNotIn("input", output)
    self.assertNotIn("chosen", output)
    self.assertNotIn("rejected", output)

    # Check that new keys exist
    self.assertIn("prompt_ids", output)
    self.assertIn("chosen_ids", output)
    self.assertIn("rejected_ids", output)
    self.assertIn("prompt_mask", output)
    self.assertIn("chosen_mask", output)
    self.assertIn("rejected_mask", output)

  def test_two_column_prefix_extraction(self):
    """Verify common prefix extraction for 2-column datasets."""
    # The column names will be remappend into "chosen" and "rejected"
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=21,
        data_column_names=("liked", "disliked"),
    )
    sample = {
        "liked": np.array([1, 2, 3, 10, 11]),
        "disliked": np.array([1, 2, 3, 20, 21, 22]),
    }
    output = prep.map(sample)

    # Prefix is [1, 2, 3], left-padded.
    self.assertEqual(output["prompt_ids"].shape[0], 10)
    np.testing.assert_array_equal(output["prompt_ids"], [self.pad_id] * 7 + [1, 2, 3])
    # Prompt mask for [0, 0, 0, 0, 0, 1, 2, 3] should be [0, 0, 0, 0, 0, 1, 1, 1]
    np.testing.assert_array_equal(output["prompt_mask"], [0] * 7 + [1, 1, 1])

    # chosen_ids (len 11) should be right-padded
    self.assertEqual(output["chosen_ids"].shape[0], 11)
    np.testing.assert_array_equal(output["chosen_ids"], [10, 11] + [self.pad_id] * 9)
    # chosen_mask for [10, 11, 0, 0, ...] should be [1, 1, 0, 0, ...]
    np.testing.assert_array_equal(output["chosen_mask"], [1, 1] + [0] * 9)

    # rejected_ids (len 11) should be right-padded
    self.assertEqual(output["rejected_ids"].shape[0], 11)
    np.testing.assert_array_equal(output["rejected_ids"], [20, 21, 22] + [self.pad_id] * 8)
    # rejected_mask for [20, 21, 22, 0, 0, ...] should be [1, 1, 1, 0, 0, ...]
    np.testing.assert_array_equal(output["rejected_mask"], [1, 1, 1] + [0] * 8)

  def test_three_column_remapping(self):
    """Verify standard 3-column remapping."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=20,
        data_column_names=("input", "chosen", "rejected"),
    )
    sample = {
        "input": np.array([1, 2]),
        "chosen": np.array([3, 4]),
        "rejected": np.array([5, 6, 7]),
    }
    output = prep.map(sample)

    self.assertNotIn("input", output)
    # Prompt should be left-padded
    np.testing.assert_array_equal(output["prompt_ids"], [self.pad_id] * 8 + [1, 2])
    np.testing.assert_array_equal(output["prompt_mask"], [0] * 8 + [1, 1])

    # Chosen and rejected are right-padded
    np.testing.assert_array_equal(output["chosen_ids"], [3, 4] + [self.pad_id] * 8)
    np.testing.assert_array_equal(output["chosen_mask"], [1, 1] + [0] * 8)
    np.testing.assert_array_equal(output["rejected_ids"], [5, 6, 7] + [self.pad_id] * 7)
    np.testing.assert_array_equal(output["rejected_mask"], [1, 1, 1] + [0] * 7)

  def test_three_column_truncation(self):
    """Verify that prompts are suffix-truncated and responses are prefix-truncated."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=9,
        data_column_names=("input", "chosen", "rejected"),
    )
    sample = {
        "input": np.arange(1, 10),
        "chosen": np.arange(10, 20),
        "rejected": np.arange(20, 30),
    }
    output = prep.map(sample)

    # Prompt is suffix-truncated to 4 chars (keeps the end).
    np.testing.assert_array_equal(output["prompt_ids"], np.arange(6, 10))
    np.testing.assert_array_equal(output["prompt_mask"], [1] * 4)

    # Chosen and rejected are prefix-truncated to 5 chars (keeps the start).
    np.testing.assert_array_equal(output["chosen_ids"], np.arange(10, 15))
    np.testing.assert_array_equal(output["chosen_mask"], [1] * 5)
    np.testing.assert_array_equal(output["rejected_ids"], np.arange(20, 25))
    np.testing.assert_array_equal(output["rejected_mask"], [1] * 5)

  def test_two_column_prefix_edge_cases(self):
    """Verify prefix extraction robustness with identical strings or prefix strings."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=20,
        data_column_names=("chosen", "rejected"),
    )

    # Case 1: Identical strings
    identical_sample = {
        "chosen": np.array([1, 2, 3]),
        "rejected": np.array([1, 2, 3]),
    }
    out_identical = prep.map(identical_sample)
    # Entire string is prefix; suffixes are empty (padded with pad_id)
    np.testing.assert_array_equal(out_identical["prompt_ids"][-3:], [1, 2, 3])
    np.testing.assert_array_equal(out_identical["chosen_ids"], [self.pad_id] * 10)
    np.testing.assert_array_equal(out_identical["rejected_ids"], [self.pad_id] * 10)

    # Case 2: One is a prefix of another
    prefix_sample = {
        "chosen": np.array([1, 2, 3]),
        "rejected": np.array([1, 2, 3, 4, 5]),
    }
    out_prefix = prep.map(prefix_sample)
    np.testing.assert_array_equal(out_prefix["prompt_ids"][-3:], [1, 2, 3])
    np.testing.assert_array_equal(out_prefix["chosen_ids"], [self.pad_id] * 10)
    np.testing.assert_array_equal(out_prefix["rejected_ids"][:2], [4, 5])

  def test_max_prompt_length_override(self):
    """Verify that max_prompt_length can be customized to adjust prompt/response split ratio."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=20,
        data_column_names=("input", "chosen", "rejected"),
        max_prompt_length=15,
    )
    sample = {
        "input": np.array([1, 2]),
        "chosen": np.array([3, 4]),
        "rejected": np.array([5, 6]),
    }
    output = prep.map(sample)

    # Prompt length should be 15, response length should be 5
    self.assertEqual(output["prompt_ids"].shape[0], 15)
    self.assertEqual(output["chosen_ids"].shape[0], 5)
    self.assertEqual(output["rejected_ids"].shape[0], 5)

  def test_missing_column_error(self):
    """Verify that a helpful error is raised when a column is missing."""
    prep = dpo_utils.DPODataFormatting(
        pad_id=self.pad_id,
        max_target_length=20,
        data_column_names=("input", "chosen", "rejected"),
    )
    # 'rejected' column is missing
    sample = {"input": np.array([1, 2]), "chosen": np.array([3, 4])}
    with self.assertRaisesRegex(KeyError, "Column 'rejected' not found in the dataset"):
      prep.map(sample)


@pytest.mark.external_training
class TestDPOPipelineProcessing(unittest.TestCase):
  """End-to-end DPO pipeline processing tests."""

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize_pydantic(
        [
            os.path.join(MAXTEXT_PKG_DIR, "dpo_trainer"),
            os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "dpo.yml"),
        ],
        per_device_batch_size=2,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"),
        train_split="train",
        enable_checkpointing=False,
        use_dpo=True,
        enable_data_shuffling=False,
        max_target_length=64,
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
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.config.tokenizer_path,
        add_bos_token=False,
        add_eos_token=False,
        legacy=False,
    )
    self.pad_id = hf_data_processing._get_pad_id(self.tokenizer)  # pylint: disable=protected-access

  def get_data_iterator(self, dataset, data_columns):
    """Helper to initialize the preprocessing pipeline."""
    return hf_data_processing.preprocessing_pipeline(
        dataloading_host_index=self.process_indices.index(jax.process_index()),
        dataloading_host_count=len(self.process_indices),
        global_mesh=self.mesh,
        dataset=dataset,
        config=self.config,
        data_column_names=data_columns,
        tokenize=self.config.tokenize_train_data,
        tokenizer_path=self.config.tokenizer_path,
        hf_access_token=self.config.hf_access_token,
        global_batch_size=self.config.global_batch_size_to_load,
        max_target_length=self.config.max_target_length,
        shuffle=self.config.enable_data_shuffling,
        data_shuffle_seed=self.config.data_shuffle_seed,
        add_bos=self.config.add_bos,
        add_eos=self.config.add_eos,
        packing=self.config.packing,
        generate_padding_batch=False,
        use_dpo=self.config.use_dpo,
        use_sft=self.config.use_sft,
        sft_train_on_completion_only=self.config.sft_train_on_completion_only,
        grain_worker_count=0,
    )

  def test_dpo_format_3_columns(self):
    """Verify that the 3-column explicit DPO dataset is processed correctly."""
    prompt_str = "Question: What is 2+2?"
    chosen_str = "Answer: 4"
    rejected_str = "Answer: 5"

    dataset = Dataset.from_dict(
        {
            "input": [prompt_str] * 10,
            "chosen": [chosen_str] * 10,
            "rejected": [rejected_str] * 10,
        }
    )
    data_iter = self.get_data_iterator(dataset, ["input", "chosen", "rejected"])
    batch = next(data_iter)

    # Verify expected keys
    for key in (
        "prompt_ids",
        "chosen_ids",
        "rejected_ids",
        "prompt_mask",
        "chosen_mask",
        "rejected_mask",
    ):
      self.assertIn(key, batch)

    # Verify batch dimensions match global batch size and split max_target_length
    max_prompt_len = self.config.max_target_length // 2
    max_response_len = self.config.max_target_length - max_prompt_len
    self.assertEqual(
        batch["prompt_ids"].shape,
        (self.config.global_batch_size_to_load, max_prompt_len),
    )
    self.assertEqual(
        batch["chosen_ids"].shape,
        (self.config.global_batch_size_to_load, max_response_len),
    )
    self.assertEqual(
        batch["rejected_ids"].shape,
        (self.config.global_batch_size_to_load, max_response_len),
    )

    # Verify decoded content directly
    decoded_prompt = self.tokenizer.decode(batch["prompt_ids"][0], skip_special_tokens=True)
    decoded_chosen = self.tokenizer.decode(batch["chosen_ids"][0], skip_special_tokens=True)
    decoded_rejected = self.tokenizer.decode(batch["rejected_ids"][0], skip_special_tokens=True)

    self.assertEqual(decoded_prompt, prompt_str)
    self.assertEqual(decoded_chosen, chosen_str)
    self.assertEqual(decoded_rejected, rejected_str)

    # Verify mask structure (left padding for prompt -> 1s at the end; right padding for responses -> 1s at start)
    self.assertEqual(batch["prompt_mask"][0][-1], 1)
    self.assertEqual(batch["chosen_mask"][0][0], 1)
    self.assertEqual(batch["rejected_mask"][0][0], 1)

  def test_dpo_format_2_columns(self):
    """Verify that 2-column DPO datasets correctly extract common prefixes."""
    # We use a clear common prefix and different suffixes
    prefix = "Common prompt context for DPO:"
    chosen_suffix = " the chosen completion"
    rejected_suffix = " the rejected completion"

    dataset = Dataset.from_dict(
        {
            "chosen": [prefix + chosen_suffix] * 10,
            "rejected": [prefix + rejected_suffix] * 10,
        }
    )
    data_iter = self.get_data_iterator(dataset, ["chosen", "rejected"])
    batch = next(data_iter)

    # Verify decoded extracted prefix and completions robustly against BPE token boundary quirks
    decoded_prompt = self.tokenizer.decode(batch["prompt_ids"][0], skip_special_tokens=True)
    decoded_chosen = self.tokenizer.decode(batch["chosen_ids"][0], skip_special_tokens=True)
    decoded_rejected = self.tokenizer.decode(batch["rejected_ids"][0], skip_special_tokens=True)

    self.assertIn("Common prompt context", decoded_prompt)
    self.assertIn("chosen", decoded_chosen)
    self.assertIn("rejected", decoded_rejected)

  def test_dpo_invalid_column_count(self):
    """Verify that passing an unsupported number of columns raises an error."""
    dataset = Dataset.from_dict({"col1": ["a"] * 10})
    with self.assertRaises((ValueError, KeyError)):
      # DPODataFormatting expects 2 or 3 columns
      data_iter = self.get_data_iterator(dataset, ["col1"])
      next(data_iter)

  def test_dpo_non_positive_max_prompt_length(self):
    """Verify that max_prompt_length <= 0 raises a validation error."""
    with self.assertRaises(ValueError):
      pyconfig.initialize_pydantic(
          [
              os.path.join(MAXTEXT_PKG_DIR, "dpo_trainer"),
              os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "dpo.yml"),
          ],
          per_device_batch_size=2,
          run_name="test",
          mesh_axes=["data"],
          logical_axis_rules=[["batch", "data"]],
          data_sharding=["data"],
          base_output_directory="gs://max-experiments/",
          tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer"),
          train_split="train",
          enable_checkpointing=False,
          use_dpo=True,
          enable_data_shuffling=False,
          max_target_length=64,
          dpo={"algo": "dpo", "max_prompt_length": 0},
      )


if __name__ == "__main__":
  unittest.main()

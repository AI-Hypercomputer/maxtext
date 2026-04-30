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

"""Unit tests for train_rl.py."""

import unittest
from unittest import mock
import grain
import pytest
from types import SimpleNamespace
import jax

from maxtext.trainers.post_train.rl import train_rl

pytestmark = [pytest.mark.post_training]
from maxtext.utils import model_creation_utils


def _get_mock_devices(devices_per_slice, num_slices=1):
  mock_devices = []
  for slice_idx in range(num_slices):
    for _ in range(devices_per_slice):
      d = mock.MagicMock()
      d.id = len(mock_devices)
      d.slice_index = slice_idx
      mock_devices.append(d)
  return mock_devices


class TrainRLTest(unittest.TestCase):
  """Tests for train_rl.py."""

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_pathways_split(self):
    """Test setup_configs_and_devices with multiple VMs and Pathways."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.5,
        sampler_devices_fraction=0.5,
    )

    # Following the pattern in distillation_checkpointing_test.py for mocking jax objects
    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch("maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic", return_value=mock_config),
    ):
      trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
          ["dummy", "dummy"]
      )

      self.assertEqual(trainer_config, mock_config)
      self.assertEqual(sampler_config, mock_config)
      self.assertEqual(len(trainer_devices), 4)
      self.assertEqual(len(sampler_devices), 4)
      self.assertEqual(trainer_devices, mock_devices[:4])
      self.assertEqual(sampler_devices, mock_devices[4:])

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_pathways_fractional_split(self):
    """Test setup_configs_and_devices with multiple VMs and custom fractions."""
    mock_devices = _get_mock_devices(8)

    mock_config = SimpleNamespace(
        num_trainer_slices=-1,
        num_samplers_slices=-1,
        chips_per_vm=4,
        use_pathways=True,
        trainer_devices_fraction=0.25,
        sampler_devices_fraction=0.75,
    )

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch("maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic", return_value=mock_config),
    ):
      _, _, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(["dummy", "dummy"])

      self.assertEqual(len(trainer_devices), 2)
      self.assertEqual(len(sampler_devices), 6)
      self.assertEqual(trainer_devices, mock_devices[:2])
      self.assertEqual(sampler_devices, mock_devices[2:])

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_multislice_not_enough_slices(self):
    """Test setup_configs_and_devices raises ValueError when not enough slices."""
    mock_devices = _get_mock_devices(num_slices=2, devices_per_slice=4)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=1,
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "Not enough slices for trainer and samplers"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"])

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_multislice_invalid_tp(self):
    """Test setup_configs_and_devices raises ValueError for invalid TP."""
    mock_devices = _get_mock_devices(num_slices=4, devices_per_slice=8)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=2,
        ici_tensor_parallelism=3,  # 8 is not divisible by 3
        ici_fsdp_parallelism=-1,
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "must be divisible by tensor parallelism"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"])

  @pytest.mark.cpu_only
  def test_setup_configs_and_devices_multislice_invalid_tp_fsdp(self):
    """Test setup_configs_and_devices raises ValueError for inconsistent TP and FSDP."""
    mock_devices = _get_mock_devices(num_slices=4, devices_per_slice=8)
    mock_config = SimpleNamespace(
        num_trainer_slices=2,
        num_samplers_slices=2,
        ici_tensor_parallelism=4,
        ici_fsdp_parallelism=3,  # 4 * 3 != 8
    )

    def side_effect(argv, **kwargs):
      res = SimpleNamespace(**vars(mock_config))
      for k, v in kwargs.items():
        setattr(res, k, v)
      return res

    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch(
            "maxtext.utils.model_creation_utils.pyconfig.initialize_pydantic",
            side_effect=side_effect,
        ),
    ):
      with self.assertRaisesRegex(ValueError, "must equal devices_per_slice"):
        model_creation_utils.setup_configs_and_devices(["dummy", "dummy"])

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_no_dp(self):
    """Test case 1: sampler_config.rollout_data_parallelism=-1 -> verify result is calculated."""
    # num_sampler_devices=16, tp=2, ep=4 -> dp should be 16 // (2 * 4) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=4,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 4,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 16), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_auto_tp(self):
    """Test case 2: dp=2, tp=-1, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_fixed_tp_dp(self):
    """Test case 3: dp=2, tp=2, num_sampler_devices=4."""
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 1,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_auto_ep(self):
    """Test case 4: ep=-1 -> verify result is calculated."""
    # num_sampler_devices=8, tp=2, dp=2 -> ep should be 8 // (2 * 2) = 2
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=-1,
    )
    expected_result = {
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    self.assertEqual(train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8), expected_result)

  @pytest.mark.cpu_only
  def test_get_rollout_kwargs_errors(self):
    """Test various error cases for get_rollout_kwargs_for_parallelism."""
    # More than one -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "At most one of .* can be -1"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * ep) != 0 when dp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=-1,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 4)

    # num_devices % (tp * dp) != 0 when ep == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=3,
        rollout_expert_parallelism=-1,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # num_devices % (dp * ep) != 0 when tp == -1
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=3,
        rollout_tensor_parallelism=-1,
        rollout_expert_parallelism=2,
    )
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

    # tp * dp * ep != num_sampler_devices when all are positive
    sampler_config = SimpleNamespace(
        rollout_data_parallelism=2,
        rollout_tensor_parallelism=2,
        rollout_expert_parallelism=1,
    )
    with self.assertRaisesRegex(ValueError, r"!= len\(sampler_devices\)"):
      train_rl.get_rollout_kwargs_for_parallelism(sampler_config, 8)

  @pytest.mark.cpu_only
  def test_prompt_filtering(self):
    """Test that prompts longer than max_prefill_predict_length are filtered out."""
    # Setup mocks
    mock_tokenizer = mock.MagicMock()

    # Define tokenizer side effect
    def tokenize_side_effect(text):
      if text == "short":
        return [0] * 5
      else:
        return [0] * 15

    mock_tokenizer.tokenize.side_effect = tokenize_side_effect

    # Define dataset mock data
    train_data = [
        {"question": "short", "answer": "a1"},
        {"question": "long", "answer": "a2"},
        {"question": "short", "answer": "a3"},
        {"question": "long", "answer": "a4"},
    ]
    test_data = [{"question": "short", "answer": "a5"}, {"question": "long", "answer": "a6"}]
    train_map_ds = grain.MapDataset.source(train_data)
    test_map_ds = grain.MapDataset.source(test_data)

    def get_dataset_side_effect(config, split, data_files=None, dataset_name=None):
      if split == "train":
        return train_map_ds
      else:
        return test_map_ds

    def get_filtered_data_side_effect(dataset_name, model_tokenizer, template_config, trainer_config, x):
      return {
          "prompts": x["question"],
          "question": x["question"],
          "answer": f"[{x['answer'], x['answer']}]",
      }

    # Configs
    trainer_config = SimpleNamespace(
        debug=SimpleNamespace(rl=False),
        rl=SimpleNamespace(use_agentic_rollout=False),
        tokenizer_path="dummy_path",
        dataset_name="dummy_dataset",
        train_split="train",
        eval_split="eval",
        hf_train_files=None,
        hf_eval_files=None,
        chat_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        max_prefill_predict_length=10,
        batch_size=2,
        num_batches=2,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
    )

    with (
        mock.patch("maxtext.trainers.post_train.rl.train_rl.get_dataset", side_effect=get_dataset_side_effect),
        mock.patch("maxtext.trainers.post_train.rl.utils_rl.process_data", side_effect=get_filtered_data_side_effect),
    ):
      train_dataset, test_dataset = train_rl.prepare_datasets(trainer_config, mock_tokenizer)

      # Check filtered train dataset
      elements = list(train_dataset)
      # dataset_size = 4. Indices [0,1,2,3] are [short, long, short, long].
      # Filtered results: [short, short].
      # batch(2) will return 1 batch of 2 elements.
      self.assertEqual(len(elements), 1)
      batch = elements[0]
      self.assertEqual(len(batch["prompts"]), 2)
      for prompt in batch["prompts"]:
        self.assertEqual(prompt, "short")

      # Check filtered test dataset
      test_elements = list(test_dataset)
      # test_data indices [0,1] are [short, long].
      # num_test_batches=1, batch_size=2 -> test dataset_size = 2.
      # Filtering results: [short].
      # batch(2) will return 1 batch of 1 element.
      self.assertEqual(len(test_elements), 1)
      test_batch = test_elements[0]
      self.assertEqual(len(test_batch["prompts"]), 1)
      self.assertEqual(test_batch["prompts"][0], "short")

  @pytest.mark.cpu_only
  @mock.patch("datasets.load_dataset")
  def test_prepare_datasets_with_split(self, mock_load):
    mock_ds = mock.MagicMock()
    mock_split_result = {
        "train": [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
        "test": [{"question": "q3", "answer": "a3"}],
    }
    mock_ds.train_test_split.return_value = mock_split_result
    mock_load.return_value = mock_ds
    mock_config = SimpleNamespace(
        debug=SimpleNamespace(rl=False),
        dataset_name="open-r1/OpenR1-Math-220k",
        eval_dataset_name="open-r1/OpenR1-Math-220k",
        train_split="train",
        hf_train_files="hf://open-r1/OpenR1-Math-220k/data/dummy.parquet",
        chat_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        num_batches=1,
        batch_size=5,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
        rl=SimpleNamespace(use_agentic_rollout=False),
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<answer>",
        solution_end_token="</answer>",
        max_prefill_predict_length=256,
    )

    train_ds, test_ds = train_rl.prepare_datasets(
        trainer_config=mock_config,
        model_tokenizer=mock.MagicMock(),
    )

    mock_load.assert_called_once_with(
        "parquet",
        data_files={mock_config.train_split: mock_config.hf_train_files},
        split=mock_config.train_split,
    )
    mock_ds.train_test_split.assert_called_once_with(test_size=0.05, seed=mock_config.data_shuffle_seed)
    train_batches, test_batches = list(train_ds), list(test_ds)
    total_train_examples = sum(len(batch["question"]) for batch in train_batches)
    assert total_train_examples == 2
    total_test_examples = sum(len(batch["question"]) for batch in test_batches)
    assert total_test_examples == 1

  @pytest.mark.cpu_only
  @mock.patch("datasets.load_dataset")
  def test_prepare_datasets_without_split(self, mock_load):
    mock_ds = mock.MagicMock()
    mock_load.return_value = mock_ds
    mock_config = SimpleNamespace(
        debug=SimpleNamespace(rl=False),
        dataset_name="openai/gsm8k",
        eval_dataset_name="openai/gsm8k",
        train_split="train",
        eval_split="test",
        hf_train_files="hf://openai/gsm8k/data/dummy.parquet",
        hf_eval_files="hf://openai/gsm8k/data/dummy.parquet",
        chat_template_path="maxtext/examples/chat_templates/gsm8k_rl.json",
        data_shuffle_seed=42,
        num_batches=1,
        batch_size=5,
        train_fraction=1.0,
        num_epoch=1,
        num_test_batches=1,
        test_batch_start_index=0,
        rl=SimpleNamespace(use_agentic_rollout=False),
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<answer>",
        solution_end_token="</answer>",
        max_prefill_predict_length=256,
    )

    _, _ = train_rl.prepare_datasets(
        trainer_config=mock_config,
        model_tokenizer=mock.MagicMock(),
    )

    expected_calls = [
        mock.call(
            "parquet",
            data_files={mock_config.train_split: mock_config.hf_train_files},
            split=mock_config.train_split,
        ),
        mock.call(
            "parquet",
            data_files={mock_config.eval_split: mock_config.hf_eval_files},
            split=mock_config.eval_split,
        ),
    ]
    mock_load.assert_has_calls(expected_calls, any_order=True)
    assert mock_load.call_count == len(expected_calls)


if __name__ == "__main__":
  unittest.main()

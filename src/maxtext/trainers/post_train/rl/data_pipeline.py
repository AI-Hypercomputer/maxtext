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

"""Dataset loading and preparation utilities for RL training."""

from __future__ import annotations
from typing import Any, Optional

import datasets
import grain
from transformers import AutoTokenizer

from maxtext.trainers.post_train.rl import utils_rl
from maxtext.input_pipeline.instruction_data_processing import load_data_template_from_file
from maxtext.utils import max_logging


def get_dataset(
    tmvp_config: Any,
    split: str = "train",
    data_files: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> grain.MapDataset:
  """Download data"""
  if data_files is None:
    data = datasets.load_dataset(dataset_name, name=tmvp_config.hf_name, split=split)
  else:  # data_files have been provided, useful for using slices of large datasets like nvidia/OpenMathInstruct-2
    data = datasets.load_dataset(
        "parquet",
        data_files={split: data_files},
        split=split,
    )
  if tmvp_config.debug.rl:
    max_logging.log(f"Loaded Hugging Face dataset {dataset_name} with split {split}. Size: {len(data)}")

  return data


def prepare_train_and_eval_dataset(
    trainer_config: Any,
    test_size: float = 0.05,
) -> dict[str, datasets.Dataset]:
  """Load and split the dataset into train and validation sets using HF's train_test_split."""
  max_logging.log(
      "WARNING: For reproducible experiments, preprocess the dataset once and "
      "define your own HfDataset subclass that directly uses the preprocessed datasets."
  )

  original_ds = get_dataset(
      trainer_config,
      split=trainer_config.train_split,
      data_files=trainer_config.hf_train_files,
      dataset_name=trainer_config.dataset_name,
  )

  if "OpenMathReasoning" in trainer_config.dataset_name:
    original_ds = original_ds.filter(lambda x: x.get("problem_type") == "has_answer_extracted")

  # Split into train and validation sets using HF's train_test_split
  split_ds = original_ds.train_test_split(test_size=test_size, seed=trainer_config.data_shuffle_seed)

  return {
      "train": split_ds["train"],
      "validation": split_ds["test"],
  }


def prepare_datasets(
    trainer_config: Any,
    model_tokenizer: AutoTokenizer,
) -> tuple[grain.IterDataset, grain.IterDataset | None]:
  """Setup and return train and test datasets."""
  template_config = load_data_template_from_file(trainer_config.chat_template_path)
  if template_config is None:
    raise ValueError(
        f"Chat template is required for processing dataset but failed to load from {trainer_config.chat_template_path}"
    )

  # Prepare train and test data from training data for certain datasets
  eval_dataset_name = getattr(trainer_config, "eval_dataset_name", None)
  test_dataset = None
  if (
      trainer_config.dataset_name
      in [
          "nvidia/OpenMathInstruct-2",
          "nvidia/OpenMathReasoning",
          "open-r1/OpenR1-Math-220k",
          "bethgelab/CuratedThoughts",
      ]
      and eval_dataset_name == trainer_config.dataset_name
  ):
    splits = prepare_train_and_eval_dataset(trainer_config)

    train_dataset = (
        grain.MapDataset.source(splits["train"])
        .shuffle(seed=trainer_config.data_shuffle_seed)
        .map(
            lambda x: utils_rl.process_data(
                trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x
            )
        )
    )

    if trainer_config.num_test_batches > 0:
      test_dataset = (
          grain.MapDataset.source(splits["validation"])
          .shuffle(seed=trainer_config.data_shuffle_seed)
          .map(
              lambda x: utils_rl.process_data(
                  trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x
              )
          )
      )
  else:
    if not eval_dataset_name:
      eval_dataset_name = trainer_config.dataset_name

    train_dataset = get_dataset(
        trainer_config,
        split=trainer_config.train_split,
        data_files=trainer_config.hf_train_files,
        dataset_name=trainer_config.dataset_name,
    )
    train_dataset = (
        grain.MapDataset.source(train_dataset)
        .shuffle(seed=trainer_config.data_shuffle_seed)
        .map(
            lambda x: utils_rl.process_data(
                trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x
            )
        )
    )

    if trainer_config.num_test_batches > 0:
      test_dataset = get_dataset(
          trainer_config,
          split=trainer_config.eval_split,
          data_files=trainer_config.hf_eval_files,
          dataset_name=eval_dataset_name,
      )
      test_dataset = (
          grain.MapDataset.source(test_dataset)
          .shuffle(seed=trainer_config.data_shuffle_seed)
          .map(lambda x: utils_rl.process_data(eval_dataset_name, model_tokenizer, template_config, trainer_config, x))
      )

  def _filter_long_prompts(x):
    tokens = model_tokenizer.tokenize(x["prompts"])
    return len(tokens) <= trainer_config.max_prefill_predict_length

  train_dataset = train_dataset.filter(_filter_long_prompts)

  # AgenticGRPOLearner uses a built in chat parser that expects raw prompts
  if getattr(trainer_config.rl, "use_agentic_rollout", False):

    def _use_raw_prompt(x):
      x["prompts"] = x["question"]
      return x

    train_dataset = train_dataset.map(_use_raw_prompt)

  dataset_size = int(trainer_config.num_batches * trainer_config.batch_size * trainer_config.train_fraction)
  train_dataset = train_dataset[:dataset_size]
  train_dataset = train_dataset.repeat(trainer_config.num_epoch)
  train_dataset = train_dataset.to_iter_dataset().batch(trainer_config.batch_size)

  if trainer_config.num_test_batches > 0:
    test_dataset = test_dataset.filter(_filter_long_prompts)
    test_dataset = test_dataset[
        trainer_config.test_batch_start_index : trainer_config.num_test_batches * trainer_config.batch_size
    ]
    test_dataset = test_dataset.to_iter_dataset().batch(trainer_config.batch_size)

  return train_dataset, test_dataset

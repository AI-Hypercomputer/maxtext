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

"""Utility functions for data processing pipelines."""

import functools

import jax
from grain.experimental import BestFitPackIterDataset, pick_performance_config
import grain.python as grain

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import tokenizer
from maxtext.utils import elastic_utils


def parse_and_keep_features(dataset, config, data_columns, tokenize):
  """Parse arrayrecord features or keep specified columns for other formats."""
  if config.grain_file_type in ("arrayrecord", "tfrecord"):
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))
  else:
    dataset = dataset.map(input_pipeline_utils.KeepFeatures(feature_names=data_columns))
  return dataset


def get_tokenizer_and_pad_id(config):
  """Builds tokenizer and extracts pad_id safely."""
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = 0
  return tokenizer_model, pad_id


def validate_and_configure_sft_columns(data_columns, tokenizer_model, chat_template=None):
  """Validates SFT data columns and configures the tokenizer chat template."""
  if chat_template and hasattr(tokenizer_model, "chat_template"):
    tokenizer_model.chat_template = chat_template

  supported_columns = [["prompt", "completion"], ["messages"], ["question", "answer"]]
  assert any(
      set(data_columns) == set(supported) for supported in supported_columns
  ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_columns}"


def get_local_batch_size(config):
  """Computes local batch size based on process count and expansion factor."""
  if config.elastic_enabled:
    batch_size = elastic_utils.get_local_batch_size(config)
  else:
    batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.expansion_factor_real_data > 1:
    # global_batch_size_to_load has been expanded in pyconfig.py when expansion_factor_real_data > 1.
    # But when using Grain, we want to keep the batch_size consistent with that in the checkpoint.
    # We revert the batch_size expansion here, but load multiple batches per step in multihost_dataloading.py.
    batch_size = int(batch_size // config.expansion_factor_real_data)
  return batch_size


def format_and_batch(dataset, config, batch_size, pad_id, data_columns, tokenizer_model, shift=True):
  """Packs or pads the dataset, batches it, and optionally shifts tokens for next-token prediction.

  When `config.grain_use_elastic_iterator` is True, batching is skipped
  (ElasticIterator performs it internally) and, if `shift=True`, the shift is
  applied pre-batch on axis 0, which is equivalent to a post-batch axis=1 shift.

  `shift` should be False for pipelines that don't do next-token prediction
  (e.g. DPO, which scores full sequences).
  """
  if config.packing:
    length_struct = {col: config.max_target_length for col in data_columns}
    max_segments = config.max_segments_per_seq
    if max_segments is not None and max_segments <= 0:
      max_segments = None
    if config.grain_packing_type == "first_fit":
      dataset = grain.experimental.FirstFitPackIterDataset(
          dataset,
          length_struct=length_struct,
          num_packing_bins=batch_size,
          max_sequences_per_bin=max_segments,
      )
    elif config.grain_packing_type == "best_fit":
      dataset = BestFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=batch_size)
    elif config.grain_packing_type == "concat_then_split":
      if config.add_bos and hasattr(tokenizer_model, "bos_id"):
        dataset = grain.experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct=length_struct,
            bos_handling=grain.experimental.BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS,
            bos_token_id=tokenizer_model.bos_id,
        )
      else:
        dataset = grain.experimental.ConcatThenSplitIterDataset(dataset, length_struct=length_struct)
    else:
      raise ValueError(f"Unknown packing type: {config.packing}")

    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
  else:
    dataset = dataset.map(input_pipeline_utils.PadOrTrimToMaxLength(config.max_target_length, pad_id))

  if config.grain_use_elastic_iterator:
    # ElasticIterator batches internally, so return the pre-batch dataset.
    if shift:
      dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=0))
    return dataset

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  if shift:
    dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=1))
  return dataset


def apply_multiprocessing_and_prefetch(dataset, config, grain_worker_count, grain_per_worker_buffer_size):
  """Applies multiprocessing and prefetching configurations to the dataset."""
  if config.grain_use_elastic_iterator:
    # ElasticIterator applies multiprocessing itself.
    return dataset
  multiprocessing_options = (
      pick_performance_config(
          ds=dataset,
          ram_budget_mb=config.grain_ram_budget_mb,
          max_workers=None,
          max_buffer_size=None,
      ).multiprocessing_options
      if grain_worker_count == -1
      else grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
  )
  return dataset.mp_prefetch(multiprocessing_options)

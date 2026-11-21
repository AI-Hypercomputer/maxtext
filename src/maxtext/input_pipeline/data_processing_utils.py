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
    dataset = dataset.map(input_pipeline_utils.KeepFeatures(feature_names=data_columns, tokenize=tokenize))
  return dataset


def get_training_objective_transform(
    config,
    *,
    shift: bool,
    use_dpo: bool,
    use_sft: bool,
    packing: bool,
    pad_id: int,
    bos_token_id: int | None,
) -> input_pipeline_utils.ShiftData | input_pipeline_utils.BlockDiffusionCorruption | None:
  """Selects target preparation for causal or block-diffusion training.

  Args:
    config: Training configuration containing the objective-specific settings.
    shift: Whether causal language-model targets should be shifted by one token.
    use_dpo: Whether the pipeline is preparing direct-preference data.
    use_sft: Whether the pipeline is preparing supervised fine-tuning data.
    packing: Whether multiple examples are packed into each sequence.
    pad_id: Token ID used to pad causal language-model examples.
    bos_token_id: Beginning-of-sequence token ID, or None when unavailable.

  Returns:
    The objective-specific Grain transform, or None when target shifting is disabled.

  Raises:
    ValueError: If the objective is unsupported or block diffusion is combined with
      DPO or packing.
  """
  objective = getattr(config, "training_objective", "causal_lm")
  if objective == "block_diffusion":
    completion_only = bool(use_sft and getattr(config, "sft_train_on_completion_only", False))
    if use_dpo:
      raise ValueError("Block-diffusion training is not compatible with DPO.")
    if packing:
      raise ValueError("Block-diffusion training requires packing=False.")
    return input_pipeline_utils.BlockDiffusionCorruption(
        block_size=config.causal_block_size,
        mask_id=config.block_diffusion_mask_id,
        min_noise=config.block_diffusion_min_noise,
        logit_alignment=config.block_diffusion_logit_alignment,
        canvas_policy=config.block_diffusion_canvas_policy,
        axis=1,
        completion_only=completion_only,
    )
  if objective != "causal_lm":
    raise ValueError(f"Unsupported training objective: {objective}")
  if shift and not use_dpo:
    return input_pipeline_utils.ShiftData(ignored_ids=[pad_id, bos_token_id], axis=1)
  return None


def get_block_diffusion_transform(config, *, use_sft, pad_id):
  """Returns the block-diffusion target transform for Grain pipelines, or None for causal LM."""
  if getattr(config, "training_objective", "causal_lm") != "block_diffusion":
    return None
  return get_training_objective_transform(
      config, shift=True, use_dpo=False, use_sft=use_sft, packing=config.packing, pad_id=pad_id, bos_token_id=None
  )


def get_tokenizer_and_pad_id(config, add_bos: bool | None = None, add_eos: bool | None = None):
  """Builds tokenizer and extracts pad_id safely."""
  bos = config.add_bos if add_bos is None else add_bos
  eos = config.add_eos if add_eos is None else add_eos
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      bos,
      eos,
      config.hf_access_token,
      trust_remote_code=getattr(config, "tokenizer_trust_remote_code", False),
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


def format_and_batch(
    dataset, config, batch_size, pad_id, data_columns, tokenizer_model, shift=True, target_transform=None
):
  """Packs or pads the dataset, batches it, and optionally shifts tokens for next-token prediction.

  When `config.grain_use_elastic_iterator` is True, batching is skipped
  (ElasticIterator performs it internally) and, if `shift=True`, the shift is
  applied pre-batch on axis 0, which is equivalent to a post-batch axis=1 shift.

  `shift` should be False for pipelines that don't do next-token prediction
  (e.g. DPO, which scores full sequences).

  `target_transform`, if given, replaces the default post-batch shift (e.g. block-diffusion corruption).
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
    if target_transform is not None:
      raise ValueError("Custom target transforms (e.g. block diffusion) are not supported with ElasticIterator.")
    # ElasticIterator batches internally, so return the pre-batch dataset.
    if shift:
      dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=0))
    return dataset

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  if target_transform is not None:
    dataset = dataset.map(target_transform)
  elif shift:
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

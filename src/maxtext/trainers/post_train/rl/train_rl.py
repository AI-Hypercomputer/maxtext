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

"""
RL Trainer

This module provides a unified `rl_train` function that consolidates the common
RL training logic. It handles model loading, reward function setup, dataset
processing, and training orchestration. By default, we run Group Relative Policy Optimization (GRPO) on
GSM8K math reasoning benchmark. The script is also flexible enough to run Group Sequence Policy Optimization (GSPO).

Usage Examples:

# GRPO on Llama3.1-8B-Instruct
python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml \
  model_name=llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=${WORKLOAD?} \
  base_output_directory=${OUTPUT_PATH?} \
  hf_access_token=${HF_TOKEN?}

# GSPO on Llama3.1-70B-Instruct
python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml \
  model_name=llama3.1-70b \
  tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=${WORKLOAD?} \
  base_output_directory=${OUTPUT_PATH?} \
  hf_access_token=${HF_TOKEN?} \
  loss_algo=gspo-token

"""

from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Optional, Sequence

import dataclasses
import datasets
import grain
import jax
import json
import logging
import os
import pathwaysutils

from absl import app
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from orbax import checkpoint as ocp
from pprint import pprint
from transformers import AutoTokenizer
import maxtext.integration.vllm.maxtext_vllm_adapter as adapter

adapter.register()
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger

os.environ["TOKENIZERS_PARALLELISM"] = "0"

from maxtext.common.common_types import DecoderBlockType
from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.configs import pyconfig, types
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout
from maxtext.trainers.post_train.rl.evaluate_rl import evaluate
from maxtext.trainers.post_train.rl import utils_rl
from maxtext.input_pipeline.instruction_data_processing import load_data_template_from_file
from maxtext.utils import max_logging, max_utils, model_creation_utils
from maxtext.utils.model_creation_utils import get_rollout_kwargs_for_parallelism

_RECURRENT_ROLLOUT_DECODER_BLOCKS = frozenset((DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5))


def rollout_prefix_caching_enabled(trainer_config: types.RLConfig) -> bool:
  """Returns whether vLLM prefix caching is safe for this rollout model.

  The MaxText vLLM adapter stores Qwen3-Next/Qwen3.5 GDN state in a
  per-request slot, outside vLLM's block-addressed prefix cache. Reusing only
  the attention KV blocks would therefore skip the cached prefix tokens while
  starting the recurrent layers from a fresh (potentially stale) state slot.
  Keep prefix caching disabled for these hybrid recurrent models until the
  adapter implements recurrent-state prefix save/restore.
  """
  enabled = bool(getattr(trainer_config, "enable_prefix_caching", False))
  if enabled and getattr(trainer_config, "decoder_block", None) in _RECURRENT_ROLLOUT_DECODER_BLOCKS:
    max_logging.log(
        "Disabling vLLM prefix caching for the hybrid recurrent rollout model: "
        "GDN state is not part of the MaxText adapter's prefix cache."
    )
    return False
  return enabled


def get_dataset(
    tmvp_config: types.RLConfig,
    split: str = "train",
    data_files: Optional[str | Sequence[str]] = None,
    dataset_name: Optional[str] = None,
) -> grain.MapDataset:
  """Download data"""
  if not data_files:
    data = datasets.load_dataset(dataset_name, name=tmvp_config.hf_subset, split=split)
  else:  # data_files have been provided, useful for using slices of large datasets or curated JSONL files
    first_file = data_files if isinstance(data_files, str) else next(iter(data_files), "")
    if str(dataset_name).lower() in ("json", "jsonl") or str(first_file).lower().endswith((".json", ".jsonl")):
      builder_name = "json"
    else:
      builder_name = "parquet"
    data = datasets.load_dataset(
        builder_name,
        data_files={split: data_files},
        split=split,
    )
  if tmvp_config.debug:
    max_logging.log(f"Loaded Hugging Face dataset {dataset_name} with split {split}. Size: {len(data)}")

  return data


def prepare_train_and_eval_dataset(
    trainer_config: types.RLConfig,
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
    trainer_config: types.RLConfig,
    model_tokenizer: AutoTokenizer,
) -> tuple[grain.IterDataset, grain.IterDataset | None]:
  """Setup and return train and test datasets."""
  _custom_processor_path = getattr(trainer_config, "dataset_processor_path", "") or ""
  template_config = (
      load_data_template_from_file(trainer_config.data_template_path)
      if getattr(trainer_config, "data_template_path", "")
      else None
  )
  if template_config is None and not _custom_processor_path:
    raise ValueError(
        f"Data template is required for processing dataset but failed to load from {trainer_config.data_template_path}"
    )

  # Optional user-provided `process_data(dataset_name, tokenizer, template, config, x) -> dict`.
  # When `dataset_processor_path` is set in config, load that file's `process_data`
  # and use it instead of the built-in utils_rl.process_data. Lets users adapt
  # custom datasets (with non-standard answer columns / cleaning) without editing maxtext.
  if _custom_processor_path:
    _process_data = utils_rl.load_custom_callable(_custom_processor_path, "process_data")
    max_logging.log(f"prepare_datasets: using custom process_data from {_custom_processor_path}")
  else:
    _process_data = utils_rl.process_data

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
        .map(lambda x: _process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x))
    )

    if trainer_config.num_test_batches > 0:
      test_dataset = (
          grain.MapDataset.source(splits["validation"])
          .shuffle(seed=trainer_config.data_shuffle_seed)
          .map(lambda x: _process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x))
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
        .map(lambda x: _process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x))
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
          .map(lambda x: _process_data(eval_dataset_name, model_tokenizer, template_config, trainer_config, x))
      )

  def _filter_long_prompts(x):
    tokens = model_tokenizer.tokenize(x["prompts"])
    return len(tokens) <= trainer_config.max_prefill_predict_length

  train_dataset = train_dataset.filter(_filter_long_prompts)

  # AgenticGRPOLearner uses a built in chat parser that expects raw prompts
  # unless a custom multi-turn environment (`env_class_path`) is configured.
  if trainer_config.rl.use_agentic_rollout and not getattr(trainer_config.rl, "env_class_path", ""):

    def _use_raw_prompt(x):
      if "question" in x:
        x["prompts"] = x["question"]
      return x

    train_dataset = train_dataset.map(_use_raw_prompt)

  dataset_size = int(trainer_config.num_batches * trainer_config.batch_size * trainer_config.train_fraction)
  train_dataset = train_dataset[:dataset_size]
  train_dataset = train_dataset.repeat(trainer_config.num_epoch)
  train_dataset = train_dataset.to_iter_dataset().batch(trainer_config.batch_size, drop_remainder=True)

  if trainer_config.num_test_batches > 0:
    # eval_batch_size = -1 (default) → use trainer_config.batch_size (legacy
    # behavior). Otherwise use the override so vLLM rollout during greedy eval
    # can pack more prompts per call — important when training batch_size is
    # small (e.g. 4 for GRPO) but the sampler has enough DP replicas to absorb
    # a much larger eval batch. Total eval examples = num_test_batches *
    # eval_batch_size_for_eval; adjust num_test_batches when changing
    # eval_batch_size to keep total eval set size constant.
    eval_batch_size_for_eval = (
        trainer_config.batch_size
        if getattr(trainer_config, "eval_batch_size", -1) <= 0
        else trainer_config.eval_batch_size
    )
    test_dataset = test_dataset.filter(_filter_long_prompts)
    test_dataset = test_dataset[
        trainer_config.test_batch_start_index : trainer_config.num_test_batches * eval_batch_size_for_eval
    ]
    test_dataset = test_dataset.to_iter_dataset().batch(eval_batch_size_for_eval, drop_remainder=True)

  return train_dataset, test_dataset


def build_reward_fns(trainer_config: types.RLConfig, make_reward_fn: Callable[..., Any]) -> list[Any]:
  """Build the reward-function stack for the RL trainer.

  `reward_functions_path` is a filesystem path to a Python file and
  `reward_functions` is a comma-separated list of function names to import from
  it. When both are set, the built-in stack is REPLACED entirely by the
  user-provided callables (so users have full control over their reward stack).
  When a custom agentic environment (`rl.env_class_path`) is configured without
  custom reward functions, uses `trajectory_reward_passthrough` so rewards from
  the environment / group postprocessor flow directly to advantage estimation.
  Otherwise the default
  (`match_format_exactly`, `match_format_approximately`, `check_numbers`) stack
  is used. Every reward function is wrapped via `make_reward_fn`.
  """
  custom_rewards_path = getattr(trainer_config, "reward_functions_path", "") or ""
  custom_rewards_names = getattr(trainer_config, "reward_functions", "") or ""
  if custom_rewards_path and custom_rewards_names:
    names = [n.strip() for n in custom_rewards_names.split(",") if n.strip()]
    reward_fns = [
        make_reward_fn(utils_rl.load_custom_callable(custom_rewards_path, n, importable_by_workers=True)) for n in names
    ]
    max_logging.log(f"reward_fns: using {len(reward_fns)} custom reward function(s) {names} from {custom_rewards_path}")
    return reward_fns
  rl_cfg = getattr(trainer_config, "rl", None)
  if rl_cfg is not None and getattr(rl_cfg, "env_class_path", ""):
    return [make_reward_fn(utils_rl.trajectory_reward_passthrough)]
  return [
      make_reward_fn(utils_rl.match_format_exactly),
      make_reward_fn(utils_rl.match_format_approximately),
      make_reward_fn(utils_rl.check_numbers),
  ]


class _RewardFn:
  """Binds `tmvp_config` to a reward function while staying picklable.

  tunix's reward worker pool (`GrpoConfig.reward_num_workers`) pickles the
  reward callable into every worker task. A nested closure cannot be pickled,
  so wrapping the reward fns in one silently forces tunix's serial fallback
  ("Reward fn ... failed in worker processes (PicklingError...)"). Instances
  of this class pickle by reference to the module-level `fn` plus the config.
  A class rather than `functools.partial` so the wrapper keeps the underlying
  fn's `__name__` (via `update_wrapper`), which tunix's reward manager uses
  for its logs and per-fn bookkeeping.
  """

  def __init__(self, fn: Callable[..., Any], tmvp_config: Any):
    functools.update_wrapper(self, fn)
    self.fn = fn
    self.tmvp_config = tmvp_config

  def __call__(self, **kwargs: Any) -> Any:
    return self.fn(tmvp_config=self.tmvp_config, **kwargs)


def _kwargs_supported_by(cls: Any, **kwargs: Any) -> dict[str, Any]:
  """Keeps only the kwargs that the installed tunix `cls` accepts.

  Lets MaxText forward newer tunix knobs while still running against an older
  pinned tunix that does not know them; dropped settings are logged. A `cls`
  that itself takes `**kwargs`, or whose signature cannot be inspected (some
  builtin/C-implemented callables), accepts anything, so every kwarg is
  passed through unchanged.
  """
  try:
    params = inspect.signature(cls).parameters
  except (ValueError, TypeError):
    return kwargs
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
    return kwargs
  dropped = sorted(k for k in kwargs if k not in params)
  if dropped:
    max_logging.log(f"Installed tunix {cls.__name__} does not support {dropped}; those settings are ignored.")
  return {k: v for k, v in kwargs.items() if k in params}


def build_metrics_logging_options(
    trainer_config: Any,
) -> metrics_logger.MetricsLoggerOptions:
  """Returns Tunix MetricsLoggerOptions configured for ML Diagnostics or defaults."""
  backends = []
  if trainer_config.managed_mldiagnostics:
    from tunix.rl.metrics import mldiag_backend  # pylint: disable=import-outside-toplevel

    backends.append(mldiag_backend.MLDiagScalarBackend)
    if trainer_config.enable_tensorboard:
      backends.append(
          functools.partial(
              metrics_logger.TensorboardBackend,
              log_dir=trainer_config.tensorboard_dir,
              flush_every_n_steps=trainer_config.log_period,
          )
      )
  return metrics_logger.MetricsLoggerOptions(
      log_dir=trainer_config.tensorboard_dir,
      flush_every_n_steps=trainer_config.log_period,
      backend_kwargs={"custom_backend": backends},
  )


def build_rl_profiler_config(
    trainer_config: Any,
) -> rl_cluster_lib.RLProfileConfig | None:
  """Returns an RLProfileConfig if xplane profiling is enabled, else None."""
  if trainer_config.profiler != "xplane":
    return None
  return rl_cluster_lib.RLProfileConfig(
      start_invocation=trainer_config.rl.profiler_start_invocation,
      num_invocations=trainer_config.rl.profiler_num_invocations,
      output_dir=trainer_config.tensorboard_dir,
      managed_mldiagnostics=trainer_config.managed_mldiagnostics,
      timeout_secs=trainer_config.rl.profiler_timeout_secs,
  )


def _resolve_mini_batch_size(trainer_config: types.RLConfig) -> int:
  """Resolves the effective policy-update mini-batch size for a rollout batch."""
  return int(trainer_config.rl.mini_batch_size or trainer_config.batch_size)


def _build_mimo_config(trainer_config: types.RLConfig) -> Any:
  """Converts MaxText `rl.mimo` config into Tunix `MimoAdvantageConfig` if present."""
  mimo_raw = trainer_config.rl.mimo
  if mimo_raw is None:
    return None
  mimo_dict = mimo_raw.model_dump() if isinstance(mimo_raw, types.MimoConfig) else dict(mimo_raw)
  try:
    from tunix.rl.algorithm_config import MimoAdvantageConfig  # pylint: disable=import-outside-toplevel

    return MimoAdvantageConfig(**_kwargs_supported_by(MimoAdvantageConfig, **mimo_dict))
  except ImportError:
    return mimo_dict


def _resolve_env_class(trainer_config: types.RLConfig) -> tuple[Any, dict[str, Any] | None]:
  """Resolves `rl.env_class_path` and `rl.env_kwargs` when configured."""
  env_class_path = trainer_config.rl.env_class_path
  if not env_class_path:
    return None, None
  env_class = utils_rl.load_custom_callable(env_class_path)
  env_kwargs = dict(trainer_config.rl.env_kwargs)
  try:
    env_params = inspect.signature(env_class).parameters
  except (ValueError, TypeError):
    env_params = {}
  accepts_enable_thinking = "enable_thinking" in env_params or any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in env_params.values()
  )
  if "enable_thinking" not in env_kwargs and (accepts_enable_thinking or trainer_config.rl.enable_thinking):
    env_kwargs["enable_thinking"] = trainer_config.rl.enable_thinking
  return env_class, env_kwargs


def _is_multi_turn_code_env(trainer_config: types.RLConfig) -> bool:
  """Returns whether a custom multi-turn agentic environment is configured."""
  return bool(trainer_config.rl.use_agentic_rollout and trainer_config.rl.env_class_path)


def _build_group_postprocessor(
    trainer_config: types.RLConfig,
    model_tokenizer: Any = None,
    mimo_cfg: Any = None,
) -> Any:
  """Resolves and instantiates `rl.group_postprocessor_path` when configured."""
  rl_cfg = trainer_config.rl
  if not rl_cfg.group_postprocessor_path:
    return None
  target = utils_rl.load_custom_callable(rl_cfg.group_postprocessor_path)
  if hasattr(target, "process_group") and not isinstance(target, type):
    return target

  if mimo_cfg is None:
    mimo_cfg = _build_mimo_config(trainer_config)
  judge_base_url = rl_cfg.judge_base_url
  judge_max_concurrency = max(64, int(rl_cfg.judge_max_concurrency))
  gar_rank_rho = rl_cfg.mimo.gar_rank_rho if rl_cfg.mimo is not None else 0.5
  max_flagged_ratio = (
      rl_cfg.mimo.max_flagged_token_ratio_per_turn
      if (
          rl_cfg.max_flagged_token_ratio_per_turn == 0.35
          and rl_cfg.mimo is not None
          and rl_cfg.mimo.max_flagged_token_ratio_per_turn != 0.35
      )
      else rl_cfg.max_flagged_token_ratio_per_turn
  )

  candidate_kwargs: dict[str, Any] = {
      "tokenizer": model_tokenizer,
      "mimo_config": mimo_cfg,
      "judge_base_url": judge_base_url,
      "judge_model": rl_cfg.judge_model,
      "local_rules_only": not bool(judge_base_url),
      "gar_rank_rho": gar_rank_rho,
      "max_faulty_segments_per_turn": rl_cfg.judge_max_flagged_segments_per_turn,
      "max_flagged_token_ratio_per_turn": max_flagged_ratio,
      "audit_only": rl_cfg.judge_audit_only,
      "max_concurrency": judge_max_concurrency,
  }
  try:
    params = inspect.signature(target).parameters
  except (ValueError, TypeError):
    params = {}
  if "judge_client" in params or "client" in params:
    try:
      from tunix.rl.agentic.environments.code import judge_client as jc_mod  # pylint: disable=import-outside-toplevel

      if judge_base_url:
        client = jc_mod.OpenAICompatJudgeClient(
            **_kwargs_supported_by(
                jc_mod.OpenAICompatJudgeClient,
                base_url=judge_base_url,
                model=rl_cfg.judge_model,
                timeout_s=rl_cfg.judge_timeout_s,
                max_concurrency=judge_max_concurrency,
                enable_thinking=rl_cfg.enable_thinking,
            )
        )
      else:
        client = jc_mod.MockJudgeClient()
      candidate_kwargs["judge_client"] = client
      if "client" in params and "judge_client" not in params:
        candidate_kwargs["client"] = client
    except ImportError:
      pass
  return target(**_kwargs_supported_by(target, **candidate_kwargs))


def _build_common_grpo_kwargs(
    trainer_config: types.RLConfig,
    mimo_cfg: Any,
) -> dict[str, Any]:
  """Returns the common kwargs shared by AgenticGrpoConfig and GrpoConfig."""
  rl_cfg = trainer_config.rl
  return {
      "num_generations": rl_cfg.num_generations,
      "num_iterations": rl_cfg.num_iterations,
      "beta": rl_cfg.grpo_beta,
      "epsilon": rl_cfg.grpo_epsilon,
      "loss_algo": rl_cfg.loss_algo,
      "loss_agg_mode": rl_cfg.loss_agg_mode,
      "epsilon_high": rl_cfg.epsilon_high,
      "use_rollout_logps": rl_cfg.use_rollout_logps,
      "gspo_pool_mode": rl_cfg.gspo_pool_mode,
      "reward_num_workers": trainer_config.reward_num_workers,
      "reward_worker_timeout_seconds": trainer_config.reward_worker_timeout_seconds,
      "advantage_estimator": rl_cfg.advantage_estimator,
      "clip_mode": rl_cfg.clip_mode,
      "epsilon_neg": rl_cfg.epsilon_neg,
      "epsilon_neg_high": rl_cfg.epsilon_neg_high,
      "token_advantage_mode": rl_cfg.token_advantage_mode,
      "mimo": mimo_cfg,
      "exact_token_continuity": rl_cfg.exact_token_continuity,
      "sampler_is": rl_cfg.sampler_is,
      "sampler_is_threshold": rl_cfg.sampler_is_threshold,
      "sampler_rs": rl_cfg.sampler_rs,
      "sampler_rs_min": rl_cfg.sampler_rs_min,
      "sampler_rs_max": rl_cfg.sampler_rs_max,
      "seq_logprob_error_threshold": rl_cfg.seq_logprob_error_threshold,
      "audit_only": rl_cfg.judge_audit_only,
      "enable_thinking": rl_cfg.enable_thinking,
      "forward_only_old_logps": rl_cfg.forward_only_old_logps,
      "max_flagged_token_ratio_per_turn": rl_cfg.max_flagged_token_ratio_per_turn,
      "gspo_length_adaptive_clipping": rl_cfg.gspo_length_adaptive_clipping,
      "gspo_ref_length": rl_cfg.gspo_ref_length,
      "gspo_clip_min": rl_cfg.gspo_clip_min,
      "gspo_clip_max": rl_cfg.gspo_clip_max,
      "min_seq_len_norm": rl_cfg.min_seq_len_norm,
  }


def build_tunix_rl_configs(
    trainer_config: types.RLConfig,
    *,
    optimizer: Any = None,
    metrics_logging_options: Any = None,
    profiler_options: Any = None,
    rl_profiler_config: Any = None,
    checkpoint_dir: str | None = None,
    checkpointing_options: Any = None,
) -> tuple[rl_cluster_lib.RLTrainingConfig, Any]:
  """Construct and validate Tunix RLTrainingConfig and GrpoConfig / AgenticGrpoConfig from MaxText RLConfig."""
  train_micro_batch_size = None if trainer_config.train_micro_batch_size == -1 else trainer_config.train_micro_batch_size
  rollout_micro_batch_size = (
      None if trainer_config.rollout_micro_batch_size == -1 else trainer_config.rollout_micro_batch_size
  )
  configured_compute_logps_mbs = trainer_config.rl.compute_logps_micro_batch_size
  if configured_compute_logps_mbs == -1:
    configured_compute_logps_mbs = None
  compute_logps_micro_batch_size = (
      configured_compute_logps_mbs if configured_compute_logps_mbs is not None else train_micro_batch_size
  )
  effective_mini_batch_size = _resolve_mini_batch_size(trainer_config)

  training_config = rl_cluster_lib.RLTrainingConfig(
      actor_optimizer=optimizer,
      eval_every_n_steps=trainer_config.eval_interval,
      max_steps=trainer_config.train_steps,
      mini_batch_size=effective_mini_batch_size,
      train_micro_batch_size=train_micro_batch_size,
      rollout_micro_batch_size=rollout_micro_batch_size,
      max_seq_token_per_tpu=trainer_config.max_seq_token_per_tpu or None,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
      checkpoint_root_directory=checkpoint_dir,
      checkpointing_options=checkpointing_options,
      **_kwargs_supported_by(
          rl_cluster_lib.RLTrainingConfig,
          rl_profiler_config=rl_profiler_config,
          compute_logps_micro_batch_size=compute_logps_micro_batch_size,
          compute_logps_chunk_size=trainer_config.rl.compute_logps_chunk_size,
      ),
  )

  mimo_cfg = _build_mimo_config(trainer_config)
  common_grpo_kwargs = _build_common_grpo_kwargs(trainer_config, mimo_cfg)
  if trainer_config.rl.use_agentic_rollout:
    try:
      from tunix.rl.agentic.agentic_grpo_learner import GrpoConfig as AgenticGrpoConfig  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ValueError(
          "tunix.rl.agentic dependencies are not installed! "
          "Please install tunix with agentic support to use 'use_agentic_rollout'."
      ) from e
    grpo_config = AgenticGrpoConfig(
        max_response_length=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
        max_concurrency=trainer_config.rl.max_concurrency,
        off_policy_steps=trainer_config.rl.off_policy_steps,
        system_prompt=trainer_config.rl.system_prompt,
        **_kwargs_supported_by(AgenticGrpoConfig, **common_grpo_kwargs),
    )
  else:
    grpo_config = GrpoConfig(
        **_kwargs_supported_by(GrpoConfig, **common_grpo_kwargs),
    )

  return training_config, grpo_config


def create_rl_components(  # pylint: disable=too-many-positional-arguments
    trainer_config: types.RLConfig,
    sampler_config: types.RLConfig,
    sampler_devices: Sequence[Any],
    actor_model: Any,
    actor_mesh: Any,
    reference_model: Any,
    reference_mesh: Any,
    rollout_mesh: Any,
    model_tokenizer: Any,
) -> tuple[Any, Any, Any, list[Any]]:
  """Setup RL cluster, trainer, and optimizer."""
  # Setup optimizer
  optimizer = utils_rl.get_optimizer(trainer_config)

  # Setup checkpointing
  if trainer_config.enable_checkpointing:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=trainer_config.checkpoint_period, max_to_keep=trainer_config.max_num_checkpoints_to_keep
    )
    checkpoint_dir = trainer_config.checkpoint_dir
  else:
    checkpointing_options = None
    checkpoint_dir = None

  # Setup metrics logging and targeted RL profiling. `profiler_options` stays
  # None: tunix `RLTrainingConfig` forbids combining the SFT step profiler with
  # `rl_profiler_config`, and only the latter traces the rollout role.
  metrics_logging_options = build_metrics_logging_options(trainer_config)
  rl_profiler_config = build_rl_profiler_config(trainer_config)

  training_config, grpo_config = build_tunix_rl_configs(
      trainer_config,
      optimizer=optimizer,
      metrics_logging_options=metrics_logging_options,
      profiler_options=None,
      rl_profiler_config=rl_profiler_config,
      checkpoint_dir=checkpoint_dir,
      checkpointing_options=checkpointing_options,
  )

  # Parse vllm_additional_config
  rollout_additional_config = None
  if trainer_config.vllm_additional_config:
    if isinstance(trainer_config.vllm_additional_config, dict):
      rollout_additional_config = trainer_config.vllm_additional_config
    elif isinstance(trainer_config.vllm_additional_config, str):
      try:
        rollout_additional_config = json.loads(trainer_config.vllm_additional_config)
      except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse additional_config JSON: {e}") from e

  # We need to parse vLLM config to get the logical axis rules for the sampler config.
  vllm_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  argv_list = ["", str(vllm_config_path), "log_config=False"]
  vllm_config = pyconfig.initialize(argv_list, config_class=types.RLConfig)

  rl_rollout_engine = functools.partial(MaxTextVllmRollout, maxtext_config=trainer_config)

  rollout_vllm_kwargs = {
      "hf_overrides": trainer_config.vllm_hf_overrides,
      "enable_expert_parallel": sampler_config.enable_expert_parallel,
      "enable_prefix_caching": rollout_prefix_caching_enabled(trainer_config),
      # Ensures vLLM model initializes with correct dtype (not float32 default)
      "dtype": trainer_config.weight_dtype.value,
  }
  if trainer_config.vllm_block_size is not None:
    # Pin the KV-cache page size; left unset, the backend derives it from the
    # engine shape, so unrelated engine changes can move it.
    rollout_vllm_kwargs["block_size"] = trainer_config.vllm_block_size

  effective_top_k = trainer_config.decode_sampling_top_k if trainer_config.decode_sampling_top_k > 0 else None

  # When grpo_beta == 0.0, create_models_and_meshes skips reference model creation and returns
  # reference_mesh=None, while Tunix's PerfMetricsExport and RLEngine still require a non-None
  # Mesh entry for Role.REFERENCE to register device timelines.
  role_to_mesh = {
      rl_cluster_lib.Role.ACTOR: actor_mesh,
      rl_cluster_lib.Role.REFERENCE: reference_mesh if reference_mesh is not None else actor_mesh,
      rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
  }
  role_to_logical_axis_rule = {
      rl_cluster_lib.Role.ACTOR: trainer_config.logical_axis_rules,
      rl_cluster_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
      rl_cluster_lib.Role.ROLLOUT: vllm_config.logical_axis_rules,
  }

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh=role_to_mesh,
      role_to_logical_axis_rule=role_to_logical_axis_rule,
      rollout_engine=rl_rollout_engine,
      offload_to_cpu=False,
      **_kwargs_supported_by(
          rl_cluster_lib.ClusterConfig,
          gc_collect_after_weight_sync=trainer_config.gc_collect_after_weight_sync,
      ),
      training_config=training_config,
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
          max_prompt_length=trainer_config.max_prefill_predict_length,
          kv_cache_size=trainer_config.max_target_length + trainer_config.kv_cache_buffer,
          temperature=trainer_config.decode_sampling_temperature,
          top_p=trainer_config.decode_sampling_nucleus_p,
          top_k=effective_top_k,
          rollout_vllm_model_version=trainer_config.tokenizer_path,
          rollout_vllm_hbm_utilization=trainer_config.hbm_utilization_vllm,
          rollout_vllm_tpu_backend_type=getattr(
              trainer_config,
              "vllm_tpu_backend_type",
              "maxtext" if "MaxText" in str(trainer_config.vllm_hf_overrides.get("architectures", [])) else "jax",
          ),
          rollout_vllm_hf_config_path=trainer_config.vllm_hf_config_path,
          rollout_vllm_additional_config=rollout_additional_config,
          rollout_vllm_init_with_random_weights=True,
          rollout_vllm_enable_dp_attention=trainer_config.enable_dp_attention,
          rollout_vllm_max_num_batched_tokens=trainer_config.max_num_batched_tokens,
          rollout_vllm_max_num_seqs=trainer_config.max_num_seqs,
          rollout_vllm_async_scheduling=trainer_config.async_scheduling,
          rollout_vllm_server_mode=trainer_config.rl.use_agentic_rollout,
          rollout_vllm_reshard_chunk_size=trainer_config.rl.reshard_chunk_size,
          rollout_vllm_kwargs=rollout_vllm_kwargs,
          rollout_vllm_sampling_kwargs={
              "stop": trainer_config.stop_strings,
              "detokenize": trainer_config.stop_strings is not None,
              "include_stop_str_in_output": trainer_config.stop_strings is not None,
          },
          # AgenticGRPOLearner requires log-probabilities from the rollout engine
          # to support off-policy filtering and multi-iteration training.
          **({"return_logprobs": True} if trainer_config.rl.use_agentic_rollout else {}),
          **get_rollout_kwargs_for_parallelism(sampler_config, len(sampler_devices)),
          **_kwargs_supported_by(
              base_rollout.RolloutConfig,
              rollout_vllm_free_kv_cache_during_weight_sync=trainer_config.free_kv_cache_during_weight_sync,
          ),
      ),
  )

  # Create RL cluster
  max_logging.log("Creating RL cluster...")
  rl_cluster_kwargs = {}
  if trainer_config.enable_tunix_perf_metrics:
    try:
      from tunix.perf import export as perf_export  # pylint: disable=import-outside-toplevel
      from tunix.perf import metrics as perf_metrics  # pylint: disable=import-outside-toplevel

      perf_config = perf_metrics.PerfMetricsConfig()
      perf_config.custom_export_fn = perf_export.PerfMetricsExport.create_metrics_export_fn(cluster_config)
      rl_cluster_kwargs["perf_config"] = perf_config
    except ImportError:
      max_logging.log(
          "enable_tunix_perf_metrics is True but tunix.perf modules are not available, skipping Tunix-managed metrics."
      )

  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor_model,
      reference=reference_model,
      tokenizer=model_tokenizer,
      cluster_config=cluster_config,
      **rl_cluster_kwargs,
  )

  def make_reward_fn(fn):
    # A picklable wrapper (not a closure) so tunix's reward worker pool can
    # ship the reward fns to its workers when `reward_num_workers` is set.
    return _RewardFn(fn, trainer_config)

  # Optional user-provided reward functions: when `reward_functions_path` and
  # `reward_functions` are both set the built-in stack is replaced entirely by
  # the user-provided callables. Each function must accept `prompts`,
  # `completions`, `tmvp_config`, and `**kwargs` and return a list of floats.
  reward_fns = build_reward_fns(trainer_config, make_reward_fn)
  mimo_cfg = getattr(grpo_config, "mimo", None) or _build_mimo_config(trainer_config)

  # Create RL trainer
  max_logging.log("Setting up RL trainer...")
  max_logging.log(
      "GRPO config resolved:\n" + "\n".join(f"  {k} = {v!r}" for k, v in sorted(dataclasses.asdict(grpo_config).items()))
  )
  if trainer_config.rl.use_agentic_rollout:
    max_logging.log("Using AgenticGRPOLearner with async online rollouts.")
    from tunix.rl.agentic.agentic_grpo_learner import GrpoLearner as AgenticGrpoLearner  # pylint: disable=import-outside-toplevel

    if _is_multi_turn_code_env(trainer_config):
      from maxtext.trainers.post_train.rl.code_rl.metrics import get_code_rl_metrics  # pylint: disable=import-outside-toplevel

      env_class, env_kwargs = _resolve_env_class(trainer_config)
      chat_parser = None
      max_resp_len = int(trainer_config.max_target_length - trainer_config.max_prefill_predict_length)
      metric_fns = [functools.partial(get_code_rl_metrics, max_response_length=max_resp_len)]
    else:
      env_class = None
      env_kwargs = None
      # Instantiate the custom MaxText chat parser
      template_config = load_data_template_from_file(trainer_config.data_template_path)
      if template_config is None:
        raise ValueError(
            f"Data template is required for AgenticGRPOLearner but failed to load from {trainer_config.data_template_path}"
        )
      chat_parser = utils_rl.MaxTextChatParser(
          model_tokenizer=model_tokenizer, template_config=template_config, tmvp_config=trainer_config
      )
      metric_fns = [utils_rl.get_correctness_metrics]

    group_postprocessor = _build_group_postprocessor(trainer_config, model_tokenizer, mimo_cfg=mimo_cfg)
    learner_extra_kwargs: dict[str, Any] = {}
    if env_class is not None:
      learner_extra_kwargs["env_class"] = env_class
      learner_extra_kwargs["env_kwargs"] = env_kwargs
    if group_postprocessor is not None:
      learner_extra_kwargs["group_postprocessor"] = group_postprocessor
      if trainer_config.rl.judge_max_in_flight_groups is not None:
        learner_extra_kwargs["group_max_in_flight"] = trainer_config.rl.judge_max_in_flight_groups
      if trainer_config.rl.judge_timeout_s is not None:
        learner_extra_kwargs["group_stage_timeout_s"] = trainer_config.rl.judge_timeout_s

    rl_trainer = AgenticGrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
        chat_parser=chat_parser,
        metric_fns=metric_fns,
        **_kwargs_supported_by(AgenticGrpoLearner, **learner_extra_kwargs),
    )
  else:
    max_logging.log("Using standard GRPOLearner with offline rollouts.")
    rl_trainer = GrpoLearner(  # pylint: disable=no-value-for-parameter
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
    )

  return rl_cluster, rl_trainer, optimizer, reward_fns


def configure_tokenizer_chat_template(model_tokenizer: Any, trainer_config: types.RLConfig) -> None:
  """Populates the tokenizer's chat_template from config if missing."""
  if getattr(model_tokenizer, "chat_template", None) is None:
    if getattr(trainer_config, "chat_template", None):
      model_tokenizer.chat_template = trainer_config.chat_template
    elif getattr(trainer_config, "chat_template_path", None):
      from maxtext.input_pipeline.instruction_data_processing import (  # pylint: disable=import-outside-toplevel
          load_chat_template_from_file,
      )

      model_tokenizer.chat_template = load_chat_template_from_file(trainer_config.chat_template_path)
    else:
      raise ValueError(
          f"Tokenizer {getattr(trainer_config, 'tokenizer_path', None)!r} has no chat_template "
          "and config.chat_template / config.chat_template_path "
          "are both empty. Either pick an instruction-tuned tokenizer that "
          "ships with a chat_template, set config.chat_template to a Jinja "
          "string, or set config.chat_template_path to a JSON file "
          "with a 'chat_template' key."
      )


def rl_train(argv: Sequence[str], kwargs: dict[str, Any]) -> None:
  """
  Run RL training with the provided configuration.

  Args:
    trainer_config: MaxText configuration for the trainer.
    sampler_config: MaxText configuration for the sampler.
    trainer_devices: JAX devices for the trainer.
    sampler_devices: JAX devices for the sampler.
  """
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
      argv,
      kwargs,
      config_class=types.RLConfig,
  )
  install_pathways_multihost_profiler_patch(trainer_config.rl.profiler_max_num_hosts)

  # Create the ML Diagnostics run before any tunix object: `RLCluster` builds
  # `RLProfiler`, whose managed backend preflights `Xprof._ensure_initialized()`
  # and raises if no MLRun is active. Both configs are uploaded (sampler keys
  # that differ from the trainer's are recorded as `sampler.<key>`).
  if trainer_config.managed_mldiagnostics:
    ManagedMLDiagnostics(trainer_config, sampler_config=sampler_config)

  goodput_recorder = create_goodput_recorder(trainer_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  _job_completed_gracefully = False
  try:
    with maybe_monitor_goodput(trainer_config):
      _rl_train_impl(trainer_config, sampler_config, trainer_devices, sampler_devices, goodput_recorder)
      _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)


def _rl_train_impl(
    trainer_config: Any, sampler_config: Any, trainer_devices: Any, sampler_devices: Any, goodput_recorder: Any = None
):
  """rl_train execution body."""
  # Create model tokenizer first so we can plumb its pad_id into the model
  # adapter (used to synthesize segment_ids that mask pad positions from
  # attention — without this the trainer attends to pad tokens and produces
  # corrupted log-probs).
  model_tokenizer = AutoTokenizer.from_pretrained(
      trainer_config.tokenizer_path,
      token=trainer_config.hf_access_token or None,
  )
  configure_tokenizer_chat_template(model_tokenizer, trainer_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = (
        model_creation_utils.create_models_and_meshes(
            trainer_config,
            sampler_config,
            trainer_devices,
            sampler_devices,
            tokenizer_pad_id=model_tokenizer.pad_token_id,
        )
    )

  if not trainer_config.debug:
    # Apply filter to suppress noisy logs
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

  if not epath.Path(trainer_config.tensorboard_dir).exists():
    epath.Path(trainer_config.tensorboard_dir).mkdir(parents=True, exist_ok=True)

  if not epath.Path(trainer_config.checkpoint_dir).exists():
    epath.Path(trainer_config.checkpoint_dir).mkdir(parents=True)

  train_dataset, test_dataset = prepare_datasets(trainer_config, model_tokenizer)

  if trainer_config.debug:
    max_logging.log("Train dataset samples:")
    for i, ele in enumerate(train_dataset):
      if i >= 5:
        break
      pprint(ele)
    if trainer_config.num_test_batches > 0:
      max_logging.log("Test dataset samples:")
      for i, ele in enumerate(test_dataset):
        if i >= 5:
          break
        pprint(ele)

  if trainer_config.debug:
    if reference_model is not None and reference_mesh is not None:
      max_logging.log("Reference Model initialized successfully")
      nnx.display(reference_model)
      max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")
    max_logging.log("Policy Model initialized successfully")
    nnx.display(actor_model)
    max_logging.log(f"Policy mesh shape: {actor_mesh.shape}")
    max_logging.log(f"Rollout_mesh shape: {rollout_mesh.shape}")

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    rl_cluster, rl_trainer, _, reward_fns = create_rl_components(
        trainer_config,
        sampler_config,
        sampler_devices,
        actor_model,
        actor_mesh,
        reference_model,
        reference_mesh,
        rollout_mesh,
        model_tokenizer,
    )

  # Run evaluation before training
  if trainer_config.num_test_batches > 0:
    # Explicitly sync actor model weights to the rollout engine before Pre-RL evaluation.
    # When resuming from an RL checkpoint (step > 0), the trainer restores RL checkpoint
    # weights into actor_model after RLCluster initializes. Without this explicit sync,
    # the rollout engine would evaluate using the base HuggingFace/SFT weights instead of
    # the restored RL checkpoint weights. Calling this unconditionally ensures weight sync
    # robustness across all initialization and restore workflows.
    rl_cluster.rollout.update_params(nnx.state(actor_model, nnx.Param))

    (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = evaluate(
        trainer_config,
        test_dataset,
        rl_cluster=rl_cluster,
        num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst,
        make_lst=trainer_config.eval_make_lst,
        reward_fns=reward_fns,
    )
    max_logging.warning(
        f"Pre RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%, {mean_reward=:.4f}"
    )

  # Start training
  if trainer_config.load_checkpoint_only_once and reference_model is not None:
    max_logging.log("Capturing reference model state before training.")
    ref_state_before = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

  # Wire intermediate eval: fire greedy `evaluate(...)` every `eval_interval`
  # outer steps. No-op when eval_interval <= 0 or num_test_batches <= 0.
  utils_rl.install_training_hooks(rl_cluster, trainer_config, test_dataset, reward_fns, goodput_recorder=goodput_recorder)

  max_logging.warning("Starting RL training...")
  rl_trainer.train(train_dataset)

  if trainer_config.load_checkpoint_only_once and reference_model is not None:
    max_logging.log("Checking if reference model state changed during training.")
    ref_state_after = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))
    check = jax.tree_util.tree_map(jax.numpy.array_equal, ref_state_before, ref_state_after)
    if not jax.tree_util.tree_all(check):
      raise ValueError("Reference model parameters changed during training!")
    max_logging.log("Reference model parameters verified to be unchanged during training.")

  max_logging.warning("RL Training Completed Successfully!")

  # Run evaluation after training
  if trainer_config.num_test_batches > 0:
    (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = evaluate(
        trainer_config,
        test_dataset,
        rl_cluster=rl_cluster,
        num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst,
        make_lst=trainer_config.eval_make_lst,
        reward_fns=reward_fns,
    )
    max_logging.warning(
        f"Post RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%, {mean_reward=:.4f}"
    )


def install_pathways_multihost_profiler_patch(max_hosts: int) -> None:
  """Injects max_num_hosts into start_trace when unset by caller on Pathways."""
  if max_hosts <= 1:
    return

  orig_start_trace = jax.profiler.start_trace
  if getattr(orig_start_trace, "_is_pathways_multihost_patched", False):
    orig_start_trace._max_hosts = max_hosts  # pylint: disable=protected-access
    return

  try:
    sig = inspect.signature(orig_start_trace)
    if "max_num_hosts" not in sig.parameters:
      max_logging.warning(
          "rl.profiler_max_num_hosts > 1 requested, but jax.profiler.start_trace "
          "does not accept max_num_hosts. Skipping monkeypatch."
      )
      return
  except (ValueError, TypeError):
    return

  @functools.wraps(orig_start_trace)
  def start_trace_wrapper(*args, **kwargs):
    if "max_num_hosts" not in kwargs and len(args) <= 4:
      kwargs["max_num_hosts"] = getattr(start_trace_wrapper, "_max_hosts", max_hosts)
    return orig_start_trace(*args, **kwargs)

  start_trace_wrapper._is_pathways_multihost_patched = True  # pylint: disable=protected-access
  start_trace_wrapper._max_hosts = max_hosts  # pylint: disable=protected-access

  jax.profiler.start_trace = start_trace_wrapper
  jax_src_profiler = getattr(getattr(jax, "_src", None), "profiler", None)
  if jax_src_profiler is not None:
    jax_src_profiler.start_trace = start_trace_wrapper


def main(argv: Sequence[str], kwargs: dict = None) -> None:
  """Main function to run RL training.

  Args:
    argv: Command-line arguments.
  """
  kwargs = kwargs or {}
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_utils.print_system_information()
  rl_train(argv, kwargs)


if __name__ == "__main__":
  app.run(main)

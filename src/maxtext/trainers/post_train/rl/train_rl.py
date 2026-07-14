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
import contextlib
import inspect
from functools import wraps
from typing import Any, Optional, Sequence

import datasets
import grain
import jax
import jax.numpy as jnp
import jax.sharding as xs
import json
import logging
import os
import pathwaysutils

try:
  import pathwaysutils.profiling as pw_prof
except ImportError:
  pw_prof = None

from absl import app
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from orbax import checkpoint as ocp
from pprint import pprint
from transformers import AutoTokenizer
import functools
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger, profiler
import tunix.generate.utils as tunix_utils


@contextlib.contextmanager
def _tpu_inference_compat_patches():
  """Tactical compat shims for tpu_inference.

  tpu_inference has two call-site assumptions that no longer hold:
    1. jax.lax.with_sharding_constraint: assumes silent reshard on mismatch,
       but current jax asserts when all mesh axes are Explicit. Fall back to
       jax.sharding.reshard on the AssertionError.
    2. tunix._apply_dtype_cast: tpu_inference JaxEinsum defaults
       param_dtype=float32 so its weights initialize as float32, but model
       dtype is bfloat16; the cast upgraded synced bfloat16 weights to float32,
       which then mismatched in the ragged paged attention kernel. Skip the
       bf16->f32 upcast so synced weights stay bfloat16.

  Scoped to rl_train() so the patches don't leak into other importers of this
  module. Drop both once tpu_inference is updated upstream.
  """
  orig_wsc = jax.lax.with_sharding_constraint
  orig_apply_dtype_cast = tunix_utils._apply_dtype_cast  # pylint: disable=protected-access

  def _compat_wsc(x, shardings):
    try:
      return orig_wsc(x, shardings)
    except AssertionError:
      return jax.sharding.reshard(x, shardings)

  def _no_bf16_to_f32_cast(val, tgt_dtype, src_key):
    if hasattr(val, "dtype") and val.dtype == jnp.bfloat16 and tgt_dtype == jnp.float32:
      return val
    return orig_apply_dtype_cast(val, tgt_dtype, src_key)

  jax.lax.with_sharding_constraint = _compat_wsc
  tunix_utils._apply_dtype_cast = _no_bf16_to_f32_cast  # pylint: disable=protected-access
  try:
    yield
  finally:
    jax.lax.with_sharding_constraint = orig_wsc
    tunix_utils._apply_dtype_cast = orig_apply_dtype_cast  # pylint: disable=protected-access


os.environ["TOKENIZERS_PARALLELISM"] = "0"

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout
from maxtext.trainers.post_train.rl.evaluate_rl import evaluate
from maxtext.trainers.post_train.rl import utils_rl
from maxtext.input_pipeline.instruction_data_processing import load_data_template_from_file
from maxtext.utils import max_logging, max_utils, model_creation_utils


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


def get_rollout_kwargs_for_parallelism(sampler_config, num_sampler_devices):
  """Get rollout kwargs for vLLM rollout when using data parallelism."""
  dp = sampler_config.rollout_data_parallelism
  tp = sampler_config.rollout_tensor_parallelism
  ep = sampler_config.rollout_expert_parallelism

  # -1 means "auto-derive from the other two". At most one can be -1.
  num_auto = sum(1 for x in [tp, dp, ep] if x == -1)
  if num_auto > 1:
    raise ValueError(
        "At most one of rollout_tensor_parallelism, rollout_data_parallelism, "
        "rollout_expert_parallelism can be -1 (auto-derived).\n"
        f"Currently resolved values:\n"
        f"  - rollout_tensor_parallelism = {tp}\n"
        f"  - rollout_data_parallelism = {dp}\n"
        f"  - rollout_expert_parallelism = {ep}\n\n"
        "To fix this, you must explicitly define at least two of these parameters in your command line arguments.\n"
        "For example, try adding 'rollout_tensor_parallelism=4' to your command."
    )

  if dp == -1:
    if num_sampler_devices % (tp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_expert_parallelism({ep}) "
          f"when rollout_data_parallelism is -1."
      )
    dp = num_sampler_devices // tp // ep
  elif tp == -1:
    if num_sampler_devices % (dp * ep) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_data_parallelism({dp}) * rollout_expert_parallelism({ep}) "
          f"when rollout_tensor_parallelism is -1."
      )
    tp = num_sampler_devices // dp // ep
  elif ep == -1:
    if num_sampler_devices % (tp * dp) != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_tensor_parallelism({tp}) * rollout_data_parallelism({dp}) "
          f"when rollout_expert_parallelism is -1."
      )
    ep = num_sampler_devices // tp // dp
  elif tp * dp * ep != num_sampler_devices:
    raise ValueError(
        f"rollout_tensor_parallelism({tp}) * "
        f"rollout_data_parallelism({dp}) * "
        f"rollout_expert_parallelism({ep}) "
        f"!= len(sampler_devices)({num_sampler_devices})"
    )

  rollout_kwargs = {}
  rollout_kwargs["tensor_parallel_size"] = tp
  rollout_kwargs["data_parallel_size"] = dp
  rollout_kwargs["expert_parallel_size"] = ep

  return rollout_kwargs


def get_max_train_steps(trainer_config):
  """Calculate the total number of training steps."""
  return int(
      trainer_config.num_batches
      * trainer_config.rl.num_iterations
      * trainer_config.train_fraction
      * trainer_config.num_epoch
  )


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

  # Optional user-provided `process_data(dataset_name, tokenizer, template, config, x) -> dict`.
  # When `dataset_processor_path` is set in config, load that file's `process_data`
  # and use it instead of the built-in utils_rl.process_data. Lets users adapt
  # custom datasets (with non-standard answer columns / cleaning) without editing maxtext.
  _custom_processor_path = getattr(trainer_config, "dataset_processor_path", "") or ""
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
    test_dataset = test_dataset.to_iter_dataset().batch(eval_batch_size_for_eval)

  return train_dataset, test_dataset


def create_rl_components(
    trainer_config,
    sampler_config,
    sampler_devices,
    actor_model,
    actor_mesh,
    reference_model,
    reference_mesh,
    rollout_mesh,
    model_tokenizer,
    max_train_steps,
):
  """Setup RL cluster, trainer, and optimizer."""
  # Setup optimizer
  optimizer = utils_rl.get_optimizer(trainer_config, max_train_steps)

  # Setup checkpointing
  if trainer_config.enable_checkpointing:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=trainer_config.checkpoint_period, max_to_keep=trainer_config.max_num_checkpoints_to_keep
    )
    checkpoint_dir = trainer_config.checkpoint_dir
  else:
    checkpointing_options = None
    checkpoint_dir = None

  # Set up micro batching
  train_micro_batch_size = None if trainer_config.train_micro_batch_size == -1 else trainer_config.train_micro_batch_size
  rollout_micro_batch_size = (
      None if trainer_config.rollout_micro_batch_size == -1 else trainer_config.rollout_micro_batch_size
  )

  # Setup metrics logging
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=os.path.join(trainer_config.tensorboard_dir, "sampler") if trainer_config.tensorboard_dir else "",
      flush_every_n_steps=trainer_config.log_period,
  )

  profiler_options = None
  if trainer_config.profiler == "xplane":
    num_hosts = jax.device_count() // trainer_config.chips_per_vm
    sig = inspect.signature(profiler.ProfilerOptions)
    profiler_kwargs = {
        "log_dir": trainer_config.tensorboard_dir,
        "skip_first_n_steps": trainer_config.skip_first_n_steps_for_profiler,
        "profiler_steps": trainer_config.profiler_steps,
        "set_profile_options": False,
    }
    if "max_num_hosts" in sig.parameters:
      profiler_kwargs["max_num_hosts"] = num_hosts
    profiler_options = profiler.ProfilerOptions(**profiler_kwargs)

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
  vllm_config = pyconfig.initialize(argv_list)

  rl_rollout_engine = (
      functools.partial(MaxTextVllmRollout, maxtext_config=trainer_config)
      if trainer_config.use_standalone_converter
      else "vllm"
  )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: actor_mesh,
          rl_cluster_lib.Role.REFERENCE: reference_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      role_to_logical_axis_rule={
          rl_cluster_lib.Role.ACTOR: trainer_config.logical_axis_rules,
          rl_cluster_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
          rl_cluster_lib.Role.ROLLOUT: vllm_config.logical_axis_rules,
      },
      rollout_engine=rl_rollout_engine,
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=trainer_config.eval_interval,
          max_steps=max_train_steps,
          mini_batch_size=trainer_config.batch_size,
          train_micro_batch_size=train_micro_batch_size,
          rollout_micro_batch_size=rollout_micro_batch_size,
          metrics_logging_options=metrics_logging_options,
          profiler_options=profiler_options,
          checkpoint_root_directory=checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
          max_prompt_length=trainer_config.max_prefill_predict_length,
          kv_cache_size=trainer_config.max_target_length + trainer_config.kv_cache_buffer,
          temperature=trainer_config.decode_sampling_temperature,
          top_p=trainer_config.decode_sampling_nucleus_p,
          top_k=trainer_config.decode_sampling_top_k,
          rollout_vllm_model_version=trainer_config.tokenizer_path,
          rollout_vllm_hbm_utilization=trainer_config.hbm_utilization_vllm,
          rollout_vllm_tpu_backend_type="jax",
          rollout_vllm_hf_config_path=trainer_config.vllm_hf_config_path,
          rollout_vllm_additional_config=rollout_additional_config,
          rollout_vllm_init_with_random_weights=trainer_config.rollout_vllm_init_with_random_weights,
          rollout_vllm_enable_dp_attention=trainer_config.enable_dp_attention,
          rollout_vllm_max_num_batched_tokens=trainer_config.max_num_batched_tokens,
          rollout_vllm_max_num_seqs=trainer_config.max_num_seqs,
          rollout_vllm_async_scheduling=trainer_config.async_scheduling,
          rollout_vllm_server_mode=trainer_config.rl.use_agentic_rollout,
          rollout_vllm_reshard_chunk_size=trainer_config.rl.reshard_chunk_size,
          rollout_vllm_kwargs={
              "hf_overrides": trainer_config.vllm_hf_overrides,
              "enable_expert_parallel": sampler_config.enable_expert_parallel,
              "enable_prefix_caching": True,  # Enable prefix caching to speed up generation for long prompts
              # Ensures vLLM model initializes with correct dtype (not float32 default)
              "dtype": trainer_config.weight_dtype,
              "enable_chunked_prefill": False,
              "kv_cache_metrics": getattr(trainer_config.rl, "kv_cache_metrics", True),
              "disable_log_stats": getattr(trainer_config.rl, "disable_log_stats", False),
          },
          rollout_vllm_sampling_kwargs={
              "stop": trainer_config.stop_strings,
              "detokenize": trainer_config.stop_strings is not None,
              "include_stop_str_in_output": trainer_config.stop_strings is not None,
          },
          rollout_vllm_server_mode_submission_threshold=trainer_config.rl.rollout_vllm_server_mode_submission_threshold,
          rollout_vllm_server_mode_submission_timeout_s=getattr(
              trainer_config.rl, "rollout_vllm_server_mode_submission_timeout_s", 5.0
          ),
          return_logprobs=trainer_config.rl.return_logprobs
          if trainer_config.rl.return_logprobs is not None
          else trainer_config.rl.use_agentic_rollout,
          **get_rollout_kwargs_for_parallelism(sampler_config, len(sampler_devices)),
      ),
  )

  # Create RL cluster
  max_logging.log("Creating RL cluster...")
  rl_cluster_kwargs = {}
  if trainer_config.enable_tunix_perf_metrics:
    try:
      from tunix.perf import export as perf_export  # pylint: disable=import-outside-toplevel
      from tunix.perf import metrics as perf_metrics  # pylint: disable=import-outside-toplevel

      try:
        from tunix.perf.experimental import export as exp_perf_export  # pylint: disable=import-outside-toplevel
      except ImportError:
        exp_perf_export = None

      perf_config = perf_metrics.PerfMetricsConfig()
      perf_config.custom_export_fn = perf_export.PerfMetricsExport.create_metrics_export_fn(cluster_config)
      if exp_perf_export is not None:
        trace_dir = os.path.join(
            trainer_config.tensorboard_dir or trainer_config.base_output_directory, "perfetto_traces"
        )
        perf_config.custom_export_fn_v2 = exp_perf_export.PerfMetricsExport.from_cluster_config(
            cluster_config=cluster_config,
            trace_dir=trace_dir,
        ).export_metrics
        max_logging.log(f"Enabled Tunix V2 Perfetto tracing to GCS path: {trace_dir}")
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
    # pragma: no cover
    @wraps(fn)
    def _reward_fn(**kwargs):
      return fn(tmvp_config=trainer_config, **kwargs)

    return _reward_fn

  reward_fns = [  # type: ignore
      make_reward_fn(utils_rl.match_format_exactly),
      make_reward_fn(utils_rl.match_format_approximately),
      make_reward_fn(utils_rl.check_numbers),
  ]

  # Create RL trainer
  max_logging.log("Setting up RL trainer...")
  if trainer_config.rl.use_agentic_rollout:
    max_logging.log("Using AgenticGRPOLearner with async online rollouts.")
    # TODO: Remove this try-except once the dependency on tunix is fixed.
    try:
      from tunix.rl.agentic.agentic_grpo_learner import GrpoConfig as AgenticGrpoConfig  # pylint: disable=import-outside-toplevel
      from tunix.rl.agentic.agentic_grpo_learner import GrpoLearner as AgenticGrpoLearner  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise ValueError(
          "tunix.rl.agentic dependencies are not installed! "
          "Please install tunix with agentic support to use 'use_agentic_rollout'."
      ) from e
    grpo_config = AgenticGrpoConfig(
        num_generations=trainer_config.rl.num_generations,
        num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta,
        epsilon=trainer_config.rl.grpo_epsilon,
        loss_algo=trainer_config.rl.loss_algo,
        max_response_length=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
        max_concurrency=trainer_config.rl.max_concurrency,
        off_policy_steps=trainer_config.rl.off_policy_steps,
        system_prompt=trainer_config.rl.system_prompt,
        degenerate_group_masking=trainer_config.rl.degenerate_group_masking,
        epsilon_high=trainer_config.rl.epsilon_high,
        use_rollout_logps=trainer_config.rl.use_rollout_logps,
    )
    # Instantiate the custom MaxText chat parser
    template_config = load_data_template_from_file(trainer_config.chat_template_path)
    if template_config is None:
      raise ValueError(
          f"Chat template is required for AgenticGRPOLearner but failed to load from {trainer_config.chat_template_path}"
      )
    chat_parser = utils_rl.MaxTextChatParser(
        model_tokenizer=model_tokenizer, template_config=template_config, tmvp_config=trainer_config
    )
    rl_trainer = AgenticGrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
        chat_parser=chat_parser,
        metric_fns=[utils_rl.get_correctness_metrics],
    )
  else:
    max_logging.log("Using standard GRPOLearner with offline rollouts.")
    grpo_config = GrpoConfig(
        num_generations=trainer_config.rl.num_generations,
        num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta,
        epsilon=trainer_config.rl.grpo_epsilon,
        loss_algo=trainer_config.rl.loss_algo,
        loss_agg_mode=trainer_config.rl.loss_agg_mode,
    )
    rl_trainer = GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
    )

  return rl_cluster, rl_trainer, optimizer, reward_fns


def rl_train(argv: Sequence[str], kwargs: dict):
  """
  Run RL training with the provided configuration.

  Args:
    trainer_config: MaxText configuration for the trainer.
    sampler_config: MaxText configuration for the sampler.
    trainer_devices: JAX devices for the trainer.
    sampler_devices: JAX devices for the sampler.
  """
  try:
    # pylint: disable=import-outside-toplevel
    import tpu_inference.models.common.pathways_dummy_loader as pdl
    import tpu_inference.models.jax.utils.weight_utils as wu

    class JaxWrapper:

      def __getattr__(self, name):
        if name == "clear_caches":
          return lambda: None
        return getattr(jax, name)

    wu.jax = JaxWrapper()
    max_logging.log("Monkey-patched weight_utils.jax to override clear_caches.")

    def optimized_create_dummy_weights_on_tpu(
        sharding,
        weight_shape,
        weight_dtype,
    ):
      key = jax.random.PRNGKey(1234)
      raw_array = jax.random.uniform(
          key,
          shape=weight_shape,
          dtype=weight_dtype,
          minval=-1e-3,
          maxval=1e-3,
      )
      return jax.device_put(raw_array, sharding)

    def optimized_assign_and_shard_param(
        jax_param,
        jax_weight,
        param_name="Unknown",
        mesh=None,
    ):
      spec = jax_param.get_metadata().get("sharding", ())
      if isinstance(spec, jax.sharding.NamedSharding):
        spec = spec.spec
      elif isinstance(spec, jax.sharding.SingleDeviceSharding):
        spec = ()
      param_mesh = jax_param.get_metadata().get("mesh") or mesh
      try:
        jax_param.value = wu.shard_put(jax_weight, spec, mesh=param_mesh)
        jax_param.set_metadata("_is_loaded", True)
        del jax_weight
      except Exception as e:
        raise RuntimeError(
            f"Failed to load weight '{param_name}' with shape {jax_weight.shape} "
            f"into param with shape {jax_param.value.shape}"
        ) from e

    pdl.create_dummy_weights_on_tpu = optimized_create_dummy_weights_on_tpu
    wu.assign_and_shard_param = optimized_assign_and_shard_param
    max_logging.log(
        "Successfully monkey-patched Pathways dummy weight loader and assigner to bypass sequential JIT compilations."
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Failed to monkey-patch Pathways dummy weight loader/assigner: {e}")

  with _tpu_inference_compat_patches():
    _rl_train_impl(argv, kwargs)


def wrap_generate_with_profiling(rl_cluster, trainer_config):
  # pylint: disable=protected-access
  """Wraps rl_cluster.generate to programmatically trigger Pathways profiling."""
  original_generate = rl_cluster.generate
  profiled_steps = set()

  @wraps(original_generate)
  def patched_generate(*args, **kwargs):
    nonlocal profiled_steps
    profiler_options = rl_cluster.cluster_config.training_config.profiler_options
    mode = kwargs.get("mode", rl_cluster_lib.Mode.TRAIN)

    should_profile = False
    if profiler_options and mode == rl_cluster_lib.Mode.TRAIN:
      start_step = profiler_options.skip_first_n_steps
      end_step = start_step + profiler_options.profiler_steps
      current_step = rl_cluster.global_steps
      if start_step <= current_step < end_step and current_step not in profiled_steps:
        should_profile = True
        profiled_steps.add(current_step)

    if should_profile:
      sampler_mesh = rl_cluster.r2m[rl_cluster_lib.Role.ROLLOUT]
      # Calculate num_hosts based on the total JAX devices and TPUs per pod.
      # This ensures we capture all hosts in the JAX cluster.
      total_chips = len(jax.devices())
      tpus_per_pod = int(trainer_config.chips_per_vm)
      num_hosts = max(1, total_chips // tpus_per_pod)

      logging.warning(
          "[AGY_PROFILE] Starting sampler profile at step %d with max_num_hosts=%d (sampler mesh size: %d, total hosts: %d)",
          rl_cluster.global_steps,
          num_hosts,
          sampler_mesh.size,
          num_hosts,
      )
      kwargs["max_generation_steps"] = 16
      logging.warning("[AGY_PROFILE] Overriding max_generation_steps to 16 for profiling")
      try:
        if pw_prof is not None:
          options = jax.profiler.ProfileOptions()
          options.host_tracer_level = profiler_options.host_tracer_level
          options.python_tracer_level = profiler_options.python_tracer_level
          options.duration_ms = 60000  # 60 seconds to cover compilation + execution
          pw_prof.start_trace(
              log_dir=profiler_options.log_dir,
              profiler_options=options,
              max_num_hosts=num_hosts,  # Profile all hosts as requested
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("[AGY_PROFILE] Failed to start sampler trace: %s", e)

    try:
      return original_generate(*args, **kwargs)
    finally:
      if should_profile:
        logging.warning(
            "[AGY_PROFILE] Stopping sampler profile at step %d",
            rl_cluster.global_steps,
        )
        try:
          sharding = xs.NamedSharding(sampler_mesh, xs.PartitionSpec())
          dummy = jax.device_put(jnp.zeros((1,)), sharding)
          dummy.block_until_ready()
          logging.warning("[AGY_PROFILE] Blocked on sampler mesh successfully.")
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("[AGY_PROFILE] Failed to block on sampler mesh: %s", e)

        try:
          if pw_prof is not None:
            pw_prof.stop_trace()
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("[AGY_PROFILE] Failed to stop sampler trace: %s", e)

  rl_cluster.generate = patched_generate


def validate_config(config):
  """Validates the configuration parameters for RL training."""
  if config.optimizer_memory_host_offload:
    raise ValueError(
        "optimizer_memory_host_offload=True is not supported on the post-training "
        "RL path because the underlying Tunix RLCluster/Trainer does not "
        "support host offloading of the optimizer state."
    )

  if config.num_vocab_tiling > 1:
    raise ValueError(
        f"Vocab Tiling is not supported with RL. "
        f"num_vocab_tiling was configured to {config.num_vocab_tiling}, but it must be 1 when running train_rl."
    )


def _rl_train_impl(argv: Sequence[str], kwargs: dict):
  """rl_train body — kept separate so _tpu_inference_compat_patches wraps it cleanly."""
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
      argv, kwargs
  )
  print(f"[AGY_DEBUG] Total global devices: {jax.devices()}")
  print(f"[AGY_DEBUG] Trainer devices count: {len(trainer_devices)}")
  trainer_details = [
      (d.id, d.device_kind, getattr(d, "host_id", None), getattr(d, "process_index", None)) for d in trainer_devices
  ]
  print(f"[AGY_DEBUG] Trainer devices details: {trainer_details}")
  print(f"[AGY_DEBUG] Sampler devices count: {len(sampler_devices)}")
  sampler_details = [
      (d.id, d.device_kind, getattr(d, "host_id", None), getattr(d, "process_index", None)) for d in sampler_devices
  ]
  print(f"[AGY_DEBUG] Sampler devices details: {sampler_details}")
  validate_config(trainer_config)

  # Create model tokenizer first so we can plumb its pad_id into the model
  # adapter (used to synthesize segment_ids that mask pad positions from
  # attention — without this the trainer attends to pad tokens and produces
  # corrupted log-probs).
  model_tokenizer = AutoTokenizer.from_pretrained(
      trainer_config.tokenizer_path,
      token=trainer_config.hf_access_token or None,
  )

  reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = model_creation_utils.create_models_and_meshes(
      trainer_config,
      sampler_config,
      trainer_devices,
      sampler_devices,
      tokenizer_pad_id=model_tokenizer.pad_token_id,
  )

  if not trainer_config.debug.rl:
    # Apply filter to suppress noisy logs
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

  if not epath.Path(trainer_config.tensorboard_dir).exists():
    epath.Path(trainer_config.tensorboard_dir).mkdir(parents=True, exist_ok=True)

  if not epath.Path(trainer_config.checkpoint_dir).exists():
    epath.Path(trainer_config.checkpoint_dir).mkdir(parents=True)

  max_train_steps = get_max_train_steps(trainer_config)

  train_dataset, test_dataset = prepare_datasets(trainer_config, model_tokenizer)

  if trainer_config.debug.rl:
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

  if trainer_config.debug.rl:
    max_logging.log("Reference Model initialized successfully")
    nnx.display(reference_model)
    max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")
    max_logging.log("Policy Model initialized successfully")
    nnx.display(actor_model)
    max_logging.log(f"Policy mesh shape: {actor_mesh.shape}")
    max_logging.log(f"Rollout_mesh shape: {rollout_mesh.shape}")

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
      max_train_steps,
  )

  # Disable Tunix TrajectoryLogger to save host RAM during high-concurrency runs
  if hasattr(rl_trainer, "_trajectory_logger") and rl_trainer._trajectory_logger:  # pylint: disable=protected-access
    max_logging.log("Disabling Tunix TrajectoryLogger to prevent GCS Pandas OOM.")
    rl_trainer._trajectory_logger.stop()  # pylint: disable=protected-access
    rl_trainer._trajectory_logger = None  # pylint: disable=protected-access

  # Enable programmatic sampler profiling
  wrap_generate_with_profiling(rl_cluster, trainer_config)

  # Run evaluation before training
  if trainer_config.num_test_batches > 0:
    # Update vllm with model parameters from checkpoint
    rl_cluster.rollout.update_params(nnx.state(actor_model))

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
  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Capturing reference model state before training.")
    ref_state_before = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

  # Wire intermediate eval: fire greedy `evaluate(...)` every `eval_interval`
  # outer steps. No-op when eval_interval <= 0 or num_test_batches <= 0.
  utils_rl.install_training_hooks(rl_cluster, trainer_config, test_dataset, reward_fns)

  max_logging.warning("Starting RL training...")
  rl_trainer.train(train_dataset)

  if trainer_config.load_checkpoint_only_once:
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

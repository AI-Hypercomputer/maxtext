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

"""RL component building: cluster, trainer, reward functions, and training infrastructure."""

from __future__ import annotations
from typing import Any
import functools
import json
from maxtext.integration import tunix
import optax
import os

from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger, profiler

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout
from maxtext.trainers.post_train.rl import utils_rl
from maxtext.input_pipeline.instruction_data_processing import load_data_template_from_file
from maxtext.utils import max_logging


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
        "rollout_expert_parallelism can be -1 (auto-derived)."
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

  return {
      "tensor_parallel_size": tp,
      "data_parallel_size": dp,
      "expert_parallel_size": ep,
  }


def get_optimizer(tmvp_config: Any, max_train_steps: int) -> optax.GradientTransformation:
  """Function to obtain an optax optimizer, currently we use adamw."""
  schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=tmvp_config.learning_rate,
      # Linearly increase learning rate from 0. to learning_rate in the first
      # warmup_steps_fraction training steps, and then gradually decrease the
      # learning rate to 0 using cosine scheduler.
      warmup_steps=int(tmvp_config.warmup_steps_fraction * max_train_steps),
      decay_steps=max_train_steps,
      end_value=0.0,
  )

  # TODO: @mazumdera: try optimizer offloading with adamw
  # Add gradient clipping if specified
  # Grad clipping to prevent large gradients. We find this
  # important to keep KL divergence in check.
  def make_optimizer(learning_rate):
    transforms = []
    if tmvp_config.gradient_clipping_threshold > 0:
      transforms.append(optax.clip_by_global_norm(max_norm=tmvp_config.gradient_clipping_threshold))
    transforms.append(
        optax.adamw(
            learning_rate=learning_rate,
            b1=tmvp_config.adam_b1,
            b2=tmvp_config.adam_b2,
            weight_decay=tmvp_config.adam_weight_decay,
        )
    )
    return optax.chain(*transforms)

  # Wrap the entire optimizer (including gradient clipping) with
  # inject_hyperparams so opt_state.hyperparams['learning_rate'] is at the
  # top level of the state tree. This is required for tunix's peft_trainer to
  # automatically read and log the per-step learning rate.
  return optax.inject_hyperparams(make_optimizer)(learning_rate=schedule)


def build_reward_fns(trainer_config):
  return [
      functools.partial(utils_rl.match_format_exactly, tmvp_config=trainer_config),
      functools.partial(utils_rl.match_format_approximately, tmvp_config=trainer_config),
      functools.partial(utils_rl.check_numbers, tmvp_config=trainer_config),
  ]


def setup_checkpointing(trainer_config):
  checkpointing_options = None
  checkpoint_dir = None
  if trainer_config.enable_checkpointing:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=trainer_config.checkpoint_period, max_to_keep=trainer_config.max_num_checkpoints_to_keep
    )
    checkpoint_dir = trainer_config.checkpoint_dir
  return checkpointing_options, checkpoint_dir


def setup_metrics_logging(trainer_config) -> metrics_logger.MetricsLoggerOptions:
  return metrics_logger.MetricsLoggerOptions(
      log_dir=trainer_config.tensorboard_dir, flush_every_n_steps=trainer_config.log_period
  )


def setup_profiler(trainer_config) -> profiler.ProfilerOptions | None:
  profiler_options = None
  if trainer_config.profiler == "xplane":
    profiler_options = profiler.ProfilerOptions(
        log_dir=trainer_config.tensorboard_dir,
        skip_first_n_steps=trainer_config.skip_first_n_steps_for_profiler,
        profiler_steps=trainer_config.profiler_steps,
        set_profile_options=False,
    )
  return profiler_options


def parse_vllm_additional_config(trainer_config) -> dict | None:
  rollout_additional_config = None
  if trainer_config.vllm_additional_config:
    if isinstance(trainer_config.vllm_additional_config, dict):
      rollout_additional_config = trainer_config.vllm_additional_config
    elif isinstance(trainer_config.vllm_additional_config, str):
      try:
        rollout_additional_config = json.loads(trainer_config.vllm_additional_config)
      except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse additional_config JSON: {e}") from e
  return rollout_additional_config


def build_cluster_config(
    trainer_config,
    sampler_config,
    sampler_devices,
    actor_mesh,
    reference_mesh,
    rollout_mesh,
    optimizer,
    max_train_steps,
    train_micro_batch_size,
    rollout_micro_batch_size,
    metrics_logging_options,
    profiler_options,
    checkpoint_dir,
    checkpointing_options,
    rollout_additional_config,
) -> rl_cluster_lib.ClusterConfig:
  # We need to parse vLLM config to get the logical axis rules for the sampler config.
  vllm_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  argv_list = ["", str(vllm_config_path), "log_config=False"]
  vllm_config = pyconfig.initialize(argv_list)

  rl_rollout_engine = (
      functools.partial(MaxTextVllmRollout, maxtext_config=trainer_config)
      if trainer_config.use_standalone_converter
      else "vllm"
  )

  return rl_cluster_lib.ClusterConfig(
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
          rollout_vllm_init_with_random_weights=True,
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
          },
          rollout_vllm_sampling_kwargs={
              "stop": trainer_config.stop_strings,
              "detokenize": trainer_config.stop_strings is not None,
              "include_stop_str_in_output": trainer_config.stop_strings is not None,
          },
          # AgenticGRPOLearner requires log-probabilities from the rollout engine
          # to support off-policy filtering and multi-iteration training.
          **({"return_logprobs": True} if trainer_config.rl.use_agentic_rollout else {}),
          **get_rollout_kwargs_for_parallelism(sampler_config, len(sampler_devices)),
      ),
  )


def build_rl_cluster(
    trainer_config,
    cluster_config,
    actor_model,
    reference_model,
    model_tokenizer,
) -> rl_cluster_lib.RLCluster:
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

  return rl_cluster_lib.RLCluster(
      actor=actor_model,
      reference=reference_model,
      tokenizer=model_tokenizer,
      cluster_config=cluster_config,
      **rl_cluster_kwargs,
  )


def build_rl_trainer(
    trainer_config,
    rl_cluster,
    reward_fns,
    model_tokenizer,
) -> tunix.rl.grpo.grpo_learner.GrpoLearner | tunix.rl.agentic.agentic_grpo_learner.GrpoLearner:
  max_logging.log("Setting up RL trainer...")
  if trainer_config.rl.use_agentic_rollout:
    max_logging.log("Using AgenticGRPOLearner with async online rollouts.")
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
    )
    template_config = load_data_template_from_file(trainer_config.chat_template_path)
    if template_config is None:
      raise ValueError(
          f"Chat template is required for AgenticGRPOLearner but failed to load from {trainer_config.chat_template_path}"
      )
    chat_parser = utils_rl.MaxTextChatParser(
        model_tokenizer=model_tokenizer, template_config=template_config, tmvp_config=trainer_config
    )
    return AgenticGrpoLearner(
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
    )
    return GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
    )


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
  optimizer = get_optimizer(trainer_config, max_train_steps)
  checkpointing_options, checkpoint_dir = setup_checkpointing(trainer_config)

  train_micro_batch_size = None if trainer_config.train_micro_batch_size == -1 else trainer_config.train_micro_batch_size
  rollout_micro_batch_size = (
      None if trainer_config.rollout_micro_batch_size == -1 else trainer_config.rollout_micro_batch_size
  )

  metrics_logging_options = setup_metrics_logging(trainer_config)
  profiler_options = setup_profiler(trainer_config)
  rollout_additional_config = parse_vllm_additional_config(trainer_config)

  cluster_config = build_cluster_config(
      trainer_config,
      sampler_config,
      sampler_devices,
      actor_mesh,
      reference_mesh,
      rollout_mesh,
      optimizer,
      max_train_steps,
      train_micro_batch_size,
      rollout_micro_batch_size,
      metrics_logging_options,
      profiler_options,
      checkpoint_dir,
      checkpointing_options,
      rollout_additional_config,
  )

  rl_cluster = build_rl_cluster(
      trainer_config,
      cluster_config,
      actor_model,
      reference_model,
      model_tokenizer,
  )

  reward_fns = build_reward_fns(trainer_config)

  rl_trainer = build_rl_trainer(
      trainer_config,
      rl_cluster,
      reward_fns,
      model_tokenizer,
  )

  return rl_cluster, rl_trainer, optimizer

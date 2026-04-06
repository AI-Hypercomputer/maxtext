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
from functools import wraps
from typing import Sequence

import collections
import grain
import jax
import json
import logging
import os
import pathwaysutils
import tensorflow_datasets as tfds

from absl import app
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from jax.sharding import Mesh
from orbax import checkpoint as ocp
from pprint import pprint
from transformers import AutoTokenizer
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.rl.agentic.agentic_grpo_learner import GrpoConfig as AgenticGrpoConfig, GrpoLearner as AgenticGrpoLearner
from tunix.sft import metrics_logger, profiler

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.trainers.post_train.rl.evaluate_rl import evaluate
from maxtext.trainers.post_train.rl import utils_rl
from maxtext.input_pipeline.instruction_data_processing import load_template_from_file
from maxtext.utils import max_logging, max_utils, maxtext_utils, model_creation_utils


def get_maxtext_model(config, devices=None):
  """
  Load MaxText model with Tunix adapter.
  # Note: pass the path to your scanned checkpoint for 'load_parameters_path'.
  # To create a scanned checkpoint, you can use /maxtext/src/maxtext/checkpoint_conversion/to_maxtext.py and if
  # using Pathways, please set `USE_PATHWAYS=1` and use `$((1 - USE_PATHWAYS))` for storage flags:
  # export USE_PATHWAYS=1
  # python src/maxtext/checkpoint_conversion/to_maxtext.py \
  #  --model_name="gemma2-2b" \
  #  --base_output_directory="/path/to/your/output/directory" \
  #  --scan_layers=True \
  #  --checkpoint_storage_use_ocdbt=$((1 - USE_PATHWAYS)) \
  #  --checkpoint_storage_use_zarr3=$((1 - USE_PATHWAYS))
  # Please ensure that you pass the full path ending in `/0/items` for load_parameters_path to train_rl.py i.e.,
  # load_parameters_path=/path/to/your/output/directory/0/items
  """
  model, mesh = model_creation_utils.create_nnx_model(config, devices=devices)
  with mesh:
    use_no_op_mappings = "maxtext_config" in config.vllm_additional_config
    tunix_model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings)
    tunix_model.config = None
  return tunix_model, mesh


def get_dataset(
    model_tokenizer, tmvp_config, data_dir, split="train", data_files=None, dataset_name=None
) -> grain.MapDataset:
  """Download data"""
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if dataset_name is None:
    raise ValueError("dataset_name must be provided")

  if dataset_name.startswith("huggingface:"):
    import datasets  # pylint: disable=import-outside-toplevel

    if data_files is None:
      hf_dataset_name = dataset_name.replace("huggingface:", "")
      data = datasets.load_dataset(hf_dataset_name, split=split, cache_dir=data_dir)
      if tmvp_config.debug.rl:
        max_logging.log(f"Loaded Hugging Face dataset {hf_dataset_name} with split {split}. Size: {len(data)}")
    else:  # data_files have been provided, useful for using slices of large datasets like nvidia/OpenMathInstruct-2
      data = datasets.load_dataset(
          "parquet",
          data_files={tmvp_config.train_split: data_files},
          split=split,
          cache_dir=data_dir,
      )
  else:
    builder_kwargs = {"file_format": tfds.core.FileFormat.ARRAY_RECORD}
    data = tfds.data_source(
        dataset_name,
        split=split,
        data_dir=data_dir,
        builder_kwargs=builder_kwargs,
        download=True,
    )

  template_config = load_template_from_file(tmvp_config.chat_template_path)

  loaded_dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=tmvp_config.data_shuffle_seed)
      .map(lambda x: utils_rl.process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x))
  )
  return loaded_dataset


def setup_configs_and_devices(argv: list[str]):
  """Setup device allocation and configs for training and inference."""
  config = pyconfig.initialize_pydantic(argv)
  devices = jax.devices()
  if config.num_trainer_slices == -1 and config.num_samplers_slices == -1:
    max_logging.log("Running RL on a single slice")
    num_vms = len(devices) // config.chips_per_vm
    trainer_devices = devices
    sampler_devices = devices
    if num_vms >= 2 and config.use_pathways:
      # Multiple hosts with Pathways - potentially split devices for trainer and sampler
      # based on trainer_devices_fraction and sampler_devices_fraction
      max_logging.log(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
      num_devices = len(devices)
      num_trainer_devices = int(num_devices * config.trainer_devices_fraction)
      num_sampler_devices = int(num_devices * config.sampler_devices_fraction)
      trainer_devices = devices[:num_trainer_devices]
      sampler_devices = devices[num_devices - num_sampler_devices :]
      if config.trainer_devices_fraction != 1.0:
        max_logging.log(f"Using first {len(trainer_devices)} devices as Trainer devices")
      if config.sampler_devices_fraction != 1.0:
        max_logging.log(f"Using last {len(sampler_devices)} devices as Sampler devices")
    trainer_config = config
    sampler_config = config
  elif config.num_trainer_slices > 0 and config.num_samplers_slices > 0:
    max_logging.log("Running RL with Multislice")
    devices_by_slice = collections.defaultdict(list)
    for d in devices:
      devices_by_slice[d.slice_index].append(d)
    slice_indices = sorted(devices_by_slice.keys())

    if len(slice_indices) < config.num_trainer_slices + config.num_samplers_slices:
      raise ValueError("Not enough slices for trainer and samplers")

    trainer_devices = []
    for i in range(config.num_trainer_slices):
      trainer_devices.extend(devices_by_slice[slice_indices[i]])

    sampler_devices = []
    for i in range(config.num_trainer_slices, config.num_trainer_slices + config.num_samplers_slices):
      sampler_devices.extend(devices_by_slice[slice_indices[i]])

    trainer_devices_per_slice = len(trainer_devices) // config.num_trainer_slices
    trainer_fsdp = trainer_devices_per_slice
    tp = config.ici_tensor_parallelism
    if tp > 1:
      if trainer_devices_per_slice % tp != 0:
        raise ValueError(
            f"trainer_devices_per_slice ({trainer_devices_per_slice}) must be divisible by tensor parallelism ({tp})"
        )
      if config.ici_fsdp_parallelism != -1 and config.ici_fsdp_parallelism * tp != trainer_devices_per_slice:
        raise ValueError(
            f"ici_fsdp_parallelism ({config.ici_fsdp_parallelism}) * ici_tensor_parallelism ({tp}) must equal "
            f"devices_per_slice ({trainer_devices_per_slice})"
        )
      trainer_fsdp = trainer_devices_per_slice // tp

    trainer_update = {
        "num_slices": config.num_trainer_slices,
        "ici_fsdp_parallelism": trainer_fsdp,
        "ici_tensor_parallelism": tp,
        "dcn_data_parallelism": config.num_trainer_slices,
    }

    sampler_update = {
        "num_slices": config.num_samplers_slices,
        "ici_fsdp_parallelism": len(sampler_devices) // config.num_samplers_slices,
        "ici_tensor_parallelism": -1,
        "dcn_data_parallelism": config.num_samplers_slices,
    }

    trainer_config = pyconfig.initialize_pydantic(argv, **trainer_update)
    sampler_config = pyconfig.initialize_pydantic(argv, **sampler_update)

  else:
    raise ValueError("num_trainer_slices and num_samplers_slices should be both -1 or positive")

  return trainer_config, sampler_config, trainer_devices, sampler_devices


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


def prepare_datasets(trainer_config, model_tokenizer):
  """Setup and return train and test datasets."""
  home = os.path.expanduser("~") + "/"
  train_data_dir = f"{home}/data/train"
  test_data_dir = f"{home}/data/test"
  if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
  if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

  # Load datasets
  if trainer_config.dataset_name == "huggingface:nvidia/OpenMathInstruct-2":
    import datasets  # pylint: disable=import-outside-toplevel

    def prepare_openinstructmath2_dataset(
        split: str = "train_1M",
        seed: int = 42,
        test_size: float = 0.05,
    ):
      """Load and split the OpenMathInstruct-2 dataset into train and validation sets using HF's train_test_split."""
      max_logging.log(
          "WARNING: For reproducible experiments, preprocess the dataset once and "
          "define your own HfDataset subclass that directly uses the preprocessed datasets."
      )

      # Load the original dataset
      original_ds = datasets.load_dataset(
          "parquet",
          data_files={trainer_config.train_split: trainer_config.hf_train_files},
          split=split,
          cache_dir=train_data_dir,
      )

      # Split into train and validation sets using HF's train_test_split
      split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

      return {
          "train": split_ds["train"],
          "validation": split_ds["test"],
      }

    split_name = trainer_config.train_split if trainer_config.train_split != "train" else "train_1M"
    splits = prepare_openinstructmath2_dataset(split=split_name)
    template_config = load_template_from_file(trainer_config.chat_template_path)

    train_dataset = (
        grain.MapDataset.source(splits["train"])
        .shuffle(seed=trainer_config.data_shuffle_seed)
        .map(
            lambda x: utils_rl.process_data(
                trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, x
            )
        )
    )

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
    train_dataset = get_dataset(
        model_tokenizer,
        trainer_config,
        train_data_dir,
        trainer_config.train_split,
        data_files=trainer_config.hf_train_files,
        dataset_name=trainer_config.dataset_name,
    )

    eval_dataset_name = getattr(trainer_config, "eval_dataset_name", None)
    if not eval_dataset_name:
      eval_dataset_name = trainer_config.dataset_name

    test_dataset = get_dataset(
        model_tokenizer,
        trainer_config,
        test_data_dir,
        trainer_config.eval_split,
        data_files=trainer_config.hf_eval_files,
        dataset_name=eval_dataset_name,
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

  test_dataset = test_dataset.filter(_filter_long_prompts)
  test_dataset = test_dataset[
      trainer_config.test_batch_start_index : trainer_config.num_test_batches * trainer_config.batch_size
  ]

  test_dataset = test_dataset.to_iter_dataset().batch(trainer_config.batch_size)
  return train_dataset, test_dataset


def create_models_and_meshes(trainer_config, sampler_config, trainer_devices, sampler_devices):
  """Create reference and actor models and their respective meshes."""
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = get_maxtext_model(trainer_config, trainer_devices)
  devices_array = maxtext_utils.create_device_mesh(sampler_config, sampler_devices)
  rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)

  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Creating policy model by copying reference model instead of restoring from checkpoint again.")
    with reference_mesh:
      actor_base_model = nnx.clone(reference_model.base)
      use_no_op_mappings = "maxtext_config" in trainer_config.vllm_additional_config
      actor_model = TunixMaxTextAdapter(base_model=actor_base_model, use_no_op_mappings=use_no_op_mappings)
      actor_model.config = None
    actor_mesh = reference_mesh
  else:
    max_logging.log("Creating policy model with same config as reference model on trainer mesh")
    actor_model, actor_mesh = get_maxtext_model(trainer_config, trainer_devices)

  return reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh


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
      log_dir=trainer_config.tensorboard_dir, flush_every_n_steps=trainer_config.log_period
  )

  profiler_options = None
  if trainer_config.profiler == "xplane":
    profiler_options = profiler.ProfilerOptions(
        log_dir=trainer_config.tensorboard_dir,
        skip_first_n_steps=trainer_config.skip_first_n_steps_for_profiler,
        profiler_steps=trainer_config.profiler_steps,
        set_profile_options=False,
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
  vllm_config = pyconfig.initialize(argv_list)

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
      rollout_engine="vllm",
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
          rollout_vllm_swap_space_size_gb=trainer_config.swap_space_vllm_gb,
          rollout_vllm_hf_config_path=trainer_config.vllm_hf_config_path,
          rollout_vllm_additional_config=rollout_additional_config,
          rollout_vllm_init_with_random_weights=True,
          rollout_vllm_enable_dp_attention=trainer_config.enable_dp_attention,
          rollout_vllm_max_num_batched_tokens=trainer_config.max_num_batched_tokens,
          rollout_vllm_max_num_seqs=trainer_config.max_num_seqs,
          rollout_vllm_async_scheduling=trainer_config.async_scheduling,
          rollout_vllm_server_mode=trainer_config.rl.use_agentic_rollout,
          rollout_vllm_kwargs={
              "hf_overrides": trainer_config.vllm_hf_overrides,
              "enable_expert_parallel": sampler_config.rollout_expert_parallelism > 1,
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
    # pragma: no cover
    @wraps(fn)
    def _reward_fn(**kwargs):
      return fn(tmvp_config=trainer_config, **kwargs)

    return _reward_fn

  reward_fns = [  # type: ignore
      make_reward_fn(utils_rl.match_format_exactly),
      make_reward_fn(utils_rl.match_format_approximately),
      # TODO(atwigg): comment out to simplify reward and overlap with check_numbers
      make_reward_fn(utils_rl.check_answer),
      make_reward_fn(utils_rl.check_numbers),
  ]

  # Create RL trainer
  max_logging.log("Setting up RL trainer...")
  if trainer_config.rl.use_agentic_rollout:
    max_logging.log("Using AgenticGRPOLearner with async online rollouts.")
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
    # Instantiate the custom MaxText chat parser
    template_config = load_template_from_file(trainer_config.chat_template_path)
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
    )
    rl_trainer = GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
    )

  return rl_cluster, rl_trainer, optimizer


def rl_train(trainer_config, sampler_config, trainer_devices, sampler_devices):
  """
  Run RL training with the provided configuration.

  Args:
    trainer_config: MaxText configuration for the trainer.
    sampler_config: MaxText configuration for the sampler.
    trainer_devices: JAX devices for the trainer.
    sampler_devices: JAX devices for the sampler.
  """
  if not trainer_config.debug.rl:
    # Apply filter to suppress noisy logs
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)

  max_logging.log("Starting RL Training")
  if not epath.Path(trainer_config.tensorboard_dir).exists():
    epath.Path(trainer_config.tensorboard_dir).mkdir(parents=True, exist_ok=True)

  if not epath.Path(trainer_config.checkpoint_dir).exists():
    epath.Path(trainer_config.checkpoint_dir).mkdir(parents=True)

  max_train_steps = get_max_train_steps(trainer_config)

  # Create model tokenizer
  model_tokenizer = AutoTokenizer.from_pretrained(trainer_config.tokenizer_path)

  train_dataset, test_dataset = prepare_datasets(trainer_config, model_tokenizer)

  if trainer_config.debug.rl:
    for i, ele in enumerate(train_dataset):
      if i >= 5:
        break
      pprint(ele)
    for i, ele in enumerate(test_dataset):
      if i >= 5:
        break
      pprint(ele)

  reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = create_models_and_meshes(
      trainer_config, sampler_config, trainer_devices, sampler_devices
  )

  if trainer_config.debug.rl:
    max_logging.log("Reference Model initialized successfully")
    nnx.display(reference_model)
    max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")
    max_logging.log("Policy Model initialized successfully")
    nnx.display(actor_model)
    max_logging.log(f"Policy mesh shape: {actor_mesh.shape}")

  rl_cluster, rl_trainer, _ = create_rl_components(
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

  # Before we train the model, let's evaluate the model on the test set so we can
  # see the improvement post training.
  (corr, total, accuracy, partial_accuracy, format_accuracy), _ = evaluate(
      trainer_config,
      test_dataset,
      rl_cluster=rl_cluster,
      num_passes=trainer_config.num_eval_passes,
      corr_lst=trainer_config.eval_corr_lst,
      make_lst=trainer_config.eval_make_lst,
  )
  max_logging.warning(f"Pre RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")

  # Start training
  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Capturing reference model state before training.")
    ref_state_before = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

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

  # Let's evaluate our model!
  (corr, total, accuracy, partial_accuracy, format_accuracy), _ = evaluate(
      trainer_config,
      test_dataset,
      rl_cluster=rl_cluster,
      num_passes=trainer_config.num_eval_passes,
      corr_lst=trainer_config.eval_corr_lst,
      make_lst=trainer_config.eval_make_lst,
  )
  max_logging.warning(f"Post RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")


def main(argv: Sequence[str]) -> None:
  """Main function to run RL training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_utils.print_system_information()
  trainer_config, sampler_config, trainer_devices, sampler_devices = setup_configs_and_devices(argv)
  rl_train(trainer_config, sampler_config, trainer_devices, sampler_devices)


if __name__ == "__main__":
  app.run(main)

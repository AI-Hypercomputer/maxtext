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
GRPO Trainer

This module provides a unified `rl_train` function that consolidates the common
RL training logic. It handles model loading, reward function setup, dataset
processing, and training orchestration. By default, we run Group Relative Policy Optimization (GRPO) on 
GSM8K math reasoning benchmark. GRPO can enhance your model's problem-solving skills on mathematical word problems,
coding problems, etc. 

Usage:
  Usage Examples:

  # Llama3.1-8B (single host)
  python3 src/MaxText/examples/train_rl \\
    --model_name=llama3.1-8b \\
    --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN \\
    --steps=100

  # Llama3.1-70B with Pathways (multi-host)
  python3 src/MaxText/examples/train_rl \\
    --model_name=llama3.1-70b \\
    --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=gs://path/to/output \\
    --hf_access_token=$HF_TOKEN \\
    --steps=100

  # Custom dataset
  python3 src/MaxText/examples/train_rl \\
    --model_name=llama3.1-8b \\
    --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN \\
    --hf_path=custom/dataset \\
    --steps=100
"""

from pprint import pprint
from absl import app
import os
import vllm
print("vLLM module location:", vllm.__file__)
# Also check if there's a local 'vllm' directory that could be interfering
print("Is there a local vllm directory?:", os.path.isdir("vllm"))
# The following import is added to address a circular import issue in vLLM.
# By pre-importing PoolingRequestOutput, we ensure it's loaded before the
# problematic import chain is triggered, resolving the ImportError.
from vllm import PoolingRequestOutput

import sys
from typing import Sequence

import jax
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
from flax import nnx
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

import grain

import pathwaysutils

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# add the parent directory (two levels up to say ~/HOME/maxtext) to sys.path if currenlt runnig from
# ~/HOME/maxtext/MaxText/examples

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to get the project root
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger

from MaxText import max_logging, max_utils, maxtext_utils, pyconfig
from MaxText import model_creation_utils
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from MaxText.rl.evaluate_rl import evaluate
from MaxText.rl import utils_rl
from MaxText.input_pipeline.instruction_data_processing import load_template_from_file


# We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word problems.



def get_maxtext_model(config, devices=None):
  """
  Load MaxText model with Tunix adapter.
  # Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
  # To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
  """
  model, mesh = model_creation_utils.create_nnx_model(config, devices)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    tunix_model.config = None
  return tunix_model, mesh


def setup_device_allocation(tmvp_config, use_pathways: bool = False):
  """Setup device allocation for training and inference."""

  devices = jax.devices()
  num_vms = len(devices) // tmvp_config.chips_per_vm
  trainer_devices = devices
  sampler_devices = devices
  if num_vms >= 2 and use_pathways:
    # Multiple hosts with Pathways - potentially split devices for trainer and sampler
    # based on trainer_devices_fraction and sampler_devices_fraction
    print(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
    num_devices = len(devices)
    num_trainer_devices = int(num_devices * tmvp_config.trainer_devices_fraction)
    num_sampler_devices = int(num_devices * tmvp_config.sampler_devices_fraction)
    trainer_devices = devices[:num_trainer_devices]
    sampler_devices = devices[num_devices - num_sampler_devices :]
    if tmvp_config.trainer_devices_fraction!=1.0:
      print(f"Using first {len(trainer_devices)} devices as Trainer devices")
    if tmvp_config.sampler_devices_fraction != 1.0:
      print(f"Using last {len(sampler_devices)} devices as Sampler devices")
  
  return trainer_devices, sampler_devices

def get_dataset(model_tokenizer, tmvp_config, data_dir, split="train") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      tmvp_config.dataset_name,
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  template_config = load_template_from_file(tmvp_config.chat_template_path)
  loaded_dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=tmvp_config.data_shuffle_seed)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": model_tokenizer.apply_chat_template(
                  [
                      {
                          "role": "user",
                          "content": template_config["TEMPLATE"].format(
                              system_prompt=template_config["SYSTEM_PROMPT"],
                              question=x["question"].decode("utf-8"),
                          ),
                      },
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": utils_rl.extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return loaded_dataset

def rl_train(tmvp_config):
  """
  Run RL training with the provided configuration.

  Args:
    tmvp_config: MaxText configuration object
  """
  # ====== Debug flag for verbose logs ======
  DEBUG = tmvp_config.debug

  print("Starting GRPO Training")

  # Number of training steps.
  max_train_steps = int(tmvp_config.num_batches * tmvp_config.num_iterations * tmvp_config.train_fraction * tmvp_config.num_epochs)

  # ====== Data ======
  # Setup data directories
  home = os.path.expanduser("~") + "/"
  train_data_dir = f"{home}/data/train"
  test_data_dir = f"{home}/data/test"
  if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
  if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
 
  # Create model tokenizer
  model_tokenizer = AutoTokenizer.from_pretrained(tmvp_config.hf_model_name)

  # Load datasets
  dataset = get_dataset(model_tokenizer, tmvp_config, train_data_dir, tmvp_config.train_split).batch(tmvp_config.batch_size)[:tmvp_config.num_batches]

  if tmvp_config.train_fraction == 1.0:
    train_dataset = dataset.repeat(tmvp_config.num_epochs)
    val_dataset = None
  else:
    train_dataset = dataset[: int(len(dataset) * tmvp_config.train_fraction)]
    train_dataset = train_dataset.repeat(tmvp_config.num_epochs)

    val_dataset = dataset[int(len(dataset) * tmvp_config.train_fraction) :].repeat(tmvp_config.num_epochs)

  test_dataset = get_dataset(model_tokenizer, tmvp_config, test_data_dir, tmvp_config.eval_split).batch(tmvp_config.batch_size)[:tmvp_config.num_test_batches]


  # Let's see how one batch of the dataset looks like!
  if tmvp_config.debug:
    for ele in train_dataset[:1]:
      pprint(ele)


  
  # Setup device allocation
  if jax.extend.backend.get_backend().platform_version == "Pathways":
    max_logging.log("Pathways backend detected. Disabling setting profile options.")
    use_pathways = True
  else:
    use_pathways = False
  max_logging.log(f"Use Pathways: {use_pathways}")
  trainer_devices, sampler_devices = setup_device_allocation(tmvp_config, use_pathways)

  # Load reference model
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = get_maxtext_model(tmvp_config, trainer_devices)
  devices_array = maxtext_utils.create_device_mesh(tmvp_config, sampler_devices)
  # if trainer_devices=sampler_devices, then rollout_mesh=reference_mesh
  # else rollout_mesh uses sampler_devices
  rollout_mesh = Mesh(devices_array, tmvp_config.mesh_axes)
  if tmvp_config.debug:
      max_logging.log("Reference Model initialized successfully")
      nnx.display(reference_model)
      max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")

      # Sanity check that weights are loaded correctly.
      _maxtext_state_flatten = nnx.state(reference_model).flat_state()
      maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
      max_logging.log(
          f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}"
      )

  # TODO: @mazumdera: change this to use lora
  # Load policy model
  max_logging.log("Creating policy model with same config as reference model on trainer mesh")
  policy_model, policy_mesh = get_maxtext_model(tmvp_config, trainer_devices)
  actor_mesh = policy_mesh

  if tmvp_config.debug:
      max_logging.log("Policy Model initialized successfully")
      nnx.display(policy_model)
      max_logging.log(f"Policy mesh shape: {policy_mesh.shape}")

  # Setup optimizer
  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=tmvp_config.learning_rate,
          # Linearly increase learning rate from 0. to learning_rate in the first 
          # warmup_steps_fraction training steps, and then gradually decrease the 
          # learning rate to 0 using cosine scheduler.
          warmup_steps=int(tmvp_config.warmup_steps_fraction*max_train_steps),
          decay_steps=max_train_steps,
      ),
      b1=tmvp_config.adam_b1,
      b2=tmvp_config.adam_b2,
      weight_decay=tmvp_config.adam_weight_decay,
  )

  # TODO: @mazumdera: try optimizer offloading with adamw
  # Add gradient clipping if specified
  # Grad clipping to prevent large gradients. We find this
  # important to keep KL divergence in check.
  if tmvp_config.gradient_clipping_threshold > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=tmvp_config.gradient_clipping_threshold),
        optimizer,
    )

  # Setup checkpointing
  ckpt_dir = tmvp_config.checkpoint_dir
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=tmvp_config.checkpoint_period, max_to_keep=tmvp_config.max_num_checkpoints_to_keep
  )

  # Setup metrics logging
  log_dir = os.path.join(tmvp_config.tensorboard_dir, f"worker_{jax.process_index()}")

  max_logging.log(f"TensorBoard logs directory: {log_dir}")
  # Metrics logger
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=log_dir, flush_every_n_steps=tmvp_config.log_period)


  # Profiler configurations
  # TODO: xfgu@: add profiling
  profiler_options = None

  # RL Cluster config
  # Note that we use vLLM as the rollout engine.
  # and we are using Tensor Parallelism for rollout
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: actor_mesh,
          rl_cluster_lib.Role.REFERENCE: reference_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine="vllm",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=tmvp_config.eval_interval,
          max_steps=max_train_steps,
          # metrics logging
          metrics_logging_options=metrics_logging_options,
          # profiling
          profiler_options=profiler_options,
          # checkpoint saving
          checkpoint_root_directory=ckpt_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=tmvp_config.max_target_length,
          max_prompt_length=tmvp_config.max_prefill_predict_length,
          kv_cache_size=tmvp_config.max_prefill_predict_length
          + tmvp_config.max_target_length
          + tmvp_config.kv_cache_buffer,
          temperature=tmvp_config.decode_sampling_temperature,
          top_p=tmvp_config.decode_sampling_nucleus_p,
          top_k=tmvp_config.decode_sampling_top_k,      
          rollout_vllm_model_version=tmvp_config.hf_model_name,
          rollout_vllm_hbm_utilization=tmvp_config.hbm_utilization_vllm,
          rollout_vllm_tpu_backend_type="jax",
          rollout_vllm_swap_space_size_gb=tmvp_config.swap_space_vllm_gb,
      ),
  )

  # The metrics logger is now managed by the GrpoLearner, so we don't need to
  # register jax.monitoring listeners separately.
  cluster_config.training_config.metrics_logging_options = metrics_logging_options

  grpo_config = GrpoConfig(
      num_generations=tmvp_config.num_generations,
      num_iterations=tmvp_config.num_iterations,
      beta=tmvp_config.grpo_beta,
      epsilon=tmvp_config.grpo_epsilon,
      loss_algo=tmvp_config.loss_algo,
  )

  # Create RL cluster
  max_logging.log("Creating RL cluster...")
  with nn_partitioning.axis_rules(tmvp_config.logical_axis_rules):
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=reference_model,
        tokenizer=model_tokenizer,
        cluster_config=cluster_config,
    )

  # Create GRPO trainer
  max_logging.log("Setting up GRPO trainer...")
  rl_trainer = GrpoLearner(
      rl_cluster=rl_cluster,
      reward_fns=[ # type: ignore
          lambda **kwargs: utils_rl.match_format_exactly(tmvp_config=tmvp_config, **kwargs),
          lambda **kwargs: utils_rl.match_format_approximately(tmvp_config=tmvp_config, **kwargs),
          lambda **kwargs: utils_rl.check_answer(tmvp_config=tmvp_config, **kwargs),
          lambda **kwargs: utils_rl.check_numbers(tmvp_config=tmvp_config, **kwargs),
      ],
      grpo_config=grpo_config,
  )



  if tmvp_config.debug:
    # verify if vllm sampler works
    output = rl_cluster.rollout.generate(
        ["The capital of France is"],
        rollout_config=base_rollout.RolloutConfig(max_tokens_to_generate=64, temperature=0.1),
    )

    max_logging.log(f"Output: {output}")
  #
  #
  # Before we train the model, let's evaluate the model on the test set so we can
  # see the improvement post training.
  #
  (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    tmvp_config,
    test_dataset,
    rl_cluster=rl_cluster,
  )
  max_logging.log(f"Pre GRPO Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")


  # Start training
  
  max_logging.log("Starting GRPO training...")

  with reference_mesh, nn_partitioning.axis_rules(tmvp_config.logical_axis_rules):
    rl_trainer.train(train_dataset)

  max_logging.log("GRPO Training Completed Successfully!")

  # Let's evaluate our model!
  (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
      tmvp_config,
      test_dataset,
      rl_cluster=rl_cluster,
  )
  max_logging.log(f"Post GRPO Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")


def main(argv: Sequence[str]) -> None:
  """Main function to run RL training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  tmvp_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  rl_train(tmvp_config)


if __name__ == "__main__":
  app.run(main)
